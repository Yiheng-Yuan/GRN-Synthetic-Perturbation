"""Development training for an explicit GRN; no simulator truth enters fitting.

The four-weight search uses only 24/8 internally grouped training targets.
The selected setting can then be refit on all 32 registered training targets.
Only that refit's saved, locked model is evaluated on the old 16-target
development holdout. These already-inspected data are not a new blind test.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from .protocol import Condition, ReplicatedSnapshot
from .q1_graph import GraphRNAODE
from .q1_pilot_data import load_modular_q1


@dataclass(frozen=True)
class GraphConfig:
    updates: int = 4800
    conditions_per_update: int = 4
    condition_microbatch: int = 1
    cells: int = 64
    hidden: int = 8
    projections: int = 24
    step: float = .25
    learning_rate: float = .001
    lambda_distribution: float = .1
    lambda_sparse: float = 1e-4
    seed: int = 20261022
    fit_times: tuple[float, ...] = (.25, 1., 2., 4., 8., 64.)

    def __post_init__(self):
        for name in ("updates", "conditions_per_update", "condition_microbatch",
                     "cells", "hidden", "projections"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.cells < 2 or self.hidden < 2 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("cells/hidden must be >=2 and seed nonnegative")
        for name in ("step", "learning_rate", "lambda_distribution", "lambda_sparse"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.step <= .25 or self.learning_rate <= 0:
            raise ValueError("step must be in (0,.25] and learning_rate positive")
        if (not self.fit_times or any(isinstance(t, bool) or not math.isfinite(t) or t <= 0
                                      for t in self.fit_times)
                or tuple(sorted(set(self.fit_times))) != tuple(self.fit_times)):
            raise ValueError("fit_times must be increasing unique positive times")


def internal_split(observations, *, seed=20261022, validation_count=8):
    """Split targets, not cells or time rows; no response values are inspected."""
    targets = {key.target for key in observations}
    if (None in targets or type(validation_count) is not int
            or not 0 < validation_count < len(targets)):
        raise ValueError("need disjoint nonempty fit and validation targets")
    order = np.random.default_rng(seed).permutation(sorted(targets))
    validation = set(order[-validation_count:].tolist())
    return ({k: v for k, v in observations.items() if k.target not in validation},
            {k: v for k, v in observations.items() if k.target in validation})


def population_losses(pred_t, pred_c, obs_t, obs_c, raw_scale, log_scale,
                      directions, *, mean_t=None, mean_c=None):
    """Balanced arm means plus marginal/joint population distances.

    Means use all opened fitting cells when provided. The population term is
    half marginal squared Wasserstein-2 and half sliced squared Wasserstein-2
    on log1p(RNA)/fixed_scale. Projections are refreshed every update, so an
    unchanging low-dimensional projection cannot hide a permanent residual.
    There is deliberately no treated-minus-control response loss.
    """
    mean_t = obs_t.mean(-2) if mean_t is None else mean_t
    mean_c = obs_c.mean(-2) if mean_c is None else mean_c
    expression_t = ((pred_t.mean(-2) - mean_t) / raw_scale).square().mean()
    expression_c = ((pred_c.mean(-2) - mean_c) / raw_scale).square().mean()

    def distance(left, right):
        a, b = torch.log1p(left) / log_scale, torch.log1p(right) / log_scale
        marginal = (a.sort(dim=-2).values - b.sort(dim=-2).values).square().mean()
        joint = ((a @ directions).sort(dim=-2).values
                 - (b @ directions).sort(dim=-2).values).square().mean()
        return .5 * (marginal + joint)

    dist_t, dist_c = distance(pred_t, obs_t), distance(pred_c, obs_c)
    return {"expression": .5 * (expression_t + expression_c),
            "expression_treated": expression_t, "expression_control": expression_c,
            "distribution": .5 * (dist_t + dist_c)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Atomic metadata replacement; an interrupted write leaves old evidence."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    names = ("__init__.py", "q1_graph.py", "q1_graph_train.py", "q1_pilot_data.py",
             "q1_pilot_evaluation.py", "protocol.py")
    return {name: sha256(Path(__file__).with_name(name)) for name in names}


def runtime(device):
    result = {"python": platform.python_version(), "numpy": np.__version__,
              "torch": str(torch.__version__), "machine": platform.machine(),
              "device": str(device)}
    if torch.device(device).type == "cuda":
        result.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return result


def _validate_inputs(baseline, observations, config):
    baseline = np.asarray(baseline, dtype=np.float32)
    if (baseline.ndim != 2 or min(baseline.shape) < 2 or not np.isfinite(baseline).all()
            or (baseline < 0).any() or not observations):
        raise ValueError("need nonnegative finite baseline cells-by-genes and observations")
    groups = {t: [] for t in config.fit_times}
    for key, snapshot in sorted(observations.items(), key=lambda p: (p[0].time, p[0].target)):
        if (key.target is None or key.target >= baseline.shape[-1] or key.time not in groups
                or snapshot.treated.shape[-1] != baseline.shape[-1] or snapshot.fit_control is None):
            raise ValueError("fit conditions need in-bounds targets, matching genes, times and controls")
        groups[key.time].append(key)
    if (any(not keys for keys in groups.values()) or
            len({tuple((k.target, k.dose) for k in keys) for keys in groups.values()}) != 1):
        raise ValueError("each configured time must cover the same target/dose groups")
    return baseline, groups


def _input_digest(baseline, observations):
    digest = hashlib.sha256()
    for key in [None, *sorted(observations, key=lambda c: (c.time, c.target, c.dose))]:
        if key is None:
            arrays = (baseline,)
        else:
            digest.update(json.dumps([key.target, key.dose, key.time]).encode())
            arrays = (observations[key].fit_treated, observations[key].fit_control)
        for value in arrays:
            array = np.ascontiguousarray(value, dtype=np.float32)
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _save_checkpoint(path, context, model, optimizer, rng, history, device):
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"schema": "q1-graph-training-v1", "context": context,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "history": history, "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else []}, temporary)
    temporary.replace(path)


def fit_graph(baseline, observations, *, config=GraphConfig(), device="cpu",
              checkpoint_path=None, checkpoint_every=60, resume=False,
              metadata=None, on_update=None):
    """Fit without accessing third repeats, validation responses or true GRN."""
    baseline, groups = _validate_inputs(baseline, observations, config)
    if (type(checkpoint_every) is not int or checkpoint_every < 0
            or resume and checkpoint_path is None):
        raise ValueError("invalid checkpoint interval or resume path")
    path = None if checkpoint_path is None else Path(checkpoint_path)
    if path is not None and (not path.parent.is_dir() or path.exists() and not resume):
        raise ValueError("checkpoint needs an existing parent and explicit resume if present")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    start = torch.as_tensor(baseline.copy(), device=device)
    raw_scale = start.std(0, correction=0).clamp_min(1.)
    log_scale = torch.log1p(start).mean(0).clamp_min(1.)
    model = GraphRNAODE(start.shape[-1], config.hidden, state_scale=log_scale,
                        baseline_mean=start.mean(0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    context = json.loads(json.dumps({"config": asdict(config), "runtime": runtime(device),
                "sources": source_hashes(), "fit_sha256": _input_digest(baseline, observations),
                "metadata": metadata}, allow_nan=False))
    history = []
    if resume:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("schema") != "q1-graph-training-v1" or saved.get("context") != context:
            raise ValueError("incompatible checkpoint: source, data, config or runtime changed")
        history = saved["history"]
        if (not isinstance(history, list) or len(history) > config.updates or
                any(row.get("update") != i + 1 or row.get("time") != config.fit_times[i % len(config.fit_times)]
                    or any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k])
                           for k in ("loss", "expression", "distribution", "expected_edges"))
                    for i, row in enumerate(history))):
            raise ValueError("invalid checkpoint history")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.device(device).type == "cuda":
            if len(saved["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA device count changed")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    treated, control = {}, {}
    for key, snapshot in observations.items():
        treated[key] = torch.tensor(snapshot.fit_treated.reshape(-1, model.n_genes),
                                    dtype=torch.float32, device=device)
        control[key] = torch.tensor(snapshot.fit_control.reshape(-1, model.n_genes),
                                    dtype=torch.float32, device=device)
    model.train()
    for update in range(len(history), config.updates):
        duration = config.fit_times[update % len(config.fit_times)]
        pool = groups[duration]
        chosen = [pool[i] for i in rng.choice(len(pool), min(len(pool), config.conditions_per_update), replace=False)]
        directions = torch.tensor(rng.normal(size=(model.n_genes, config.projections)),
                                  dtype=torch.float32, device=device)
        directions = nn.functional.normalize(directions, dim=0)
        optimizer.zero_grad(set_to_none=True)
        totals = dict.fromkeys(("expression", "expression_treated", "expression_control", "distribution"), 0.)
        for offset in range(0, len(chosen), config.condition_microbatch):
            keys = chosen[offset:offset + config.condition_microbatch]
            count, fraction = len(keys), len(keys) / len(chosen)
            indices = torch.as_tensor(rng.integers(0, len(start), (count, config.cells)), device=device)
            initial = start[indices]
            dose = torch.zeros_like(initial)
            observed_t, observed_c, means_t, means_c = [], [], [], []
            for i, key in enumerate(keys):
                dose[i, :, key.target] = key.dose
                for source, sampled, means in ((treated, observed_t, means_t), (control, observed_c, means_c)):
                    values = source[key]
                    rows = torch.as_tensor(rng.integers(0, len(values), config.cells), device=device)
                    sampled.append(values[rows])
                    means.append(values.mean(0))
            # A single relaxed graph is fixed through time and shared by both
            # arms in this microbatch. A new graph is drawn for the next rollout.
            prediction = model.integrate(torch.cat((initial, initial)),
                torch.cat((dose, torch.zeros_like(dose))), duration, step=config.step)
            losses = population_losses(prediction[:count], prediction[count:],
                torch.stack(observed_t), torch.stack(observed_c), raw_scale, log_scale, directions,
                mean_t=torch.stack(means_t), mean_c=torch.stack(means_c))
            data_loss = losses["expression"] + config.lambda_distribution * losses["distribution"]
            if not torch.isfinite(data_loss).item():
                raise RuntimeError("nonfinite graph data loss")
            (fraction * data_loss).backward()
            for name, value in losses.items():
                totals[name] += fraction * value.detach().item()
            del prediction, losses, data_loss
        expected_edges = model.expected_edges()
        sparse = config.lambda_sparse * expected_edges
        sparse.backward()
        gradients = {name: float(torch.linalg.vector_norm(parameter.grad).detach())
                     for name, parameter in model.named_parameters() if parameter.grad is not None}
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        history.append({"update": update + 1, "time": duration,
            "sampled_targets": [key.target for key in chosen], **totals,
            "loss": totals["expression"] + config.lambda_distribution * totals["distribution"] + sparse.detach().item(),
            "weighted_distribution": config.lambda_distribution * totals["distribution"],
            "weighted_sparse": sparse.detach().item(), "expected_edges": expected_edges.detach().item(),
            "binary_edges_after_update": int(model.W.sum().item()), "gradient_norms": gradients})
        if path is not None and (update + 1 == config.updates or
                                 checkpoint_every and (update + 1) % checkpoint_every == 0):
            _save_checkpoint(path, context, model, optimizer, rng, history, device)
        if on_update is not None:
            on_update(history[-1])
    model.eval()
    return model, history


@torch.no_grad()
def predict(model, baseline, condition, *, cells=128, seed=20261022, step=.25):
    """Predict using the exact deterministic binary graph later exported."""
    model.eval()
    indices = np.random.default_rng(seed).integers(0, len(baseline), cells)
    initial = torch.as_tensor(np.asarray(baseline)[indices], dtype=torch.float32,
                              device=model.raw_H.device)
    dose = torch.zeros_like(initial)
    if condition.target is not None:
        dose[:, condition.target] = condition.dose
    paired = model.integrate(torch.stack((initial, initial)),
        torch.stack((dose, torch.zeros_like(dose))), condition.time, step=step)
    if not torch.isfinite(paired).all().item():
        raise RuntimeError("nonfinite deterministic predictions")
    return paired[0], paired[1]


@torch.no_grad()
def validation_metrics(model, baseline, observations, config):
    """Only the internal validation targets' third repeat is scored here."""
    raw_scale = torch.tensor(np.maximum(np.std(baseline, axis=0), 1.), device=model.raw_H.device)
    directions = torch.eye(model.n_genes, device=model.raw_H.device)
    rows = []
    for key, snapshot in sorted(observations.items(), key=lambda p: (p[0].time, p[0].target)):
        a, b = predict(model, baseline, key, cells=len(snapshot.treated[2]), seed=config.seed, step=config.step)
        observed = [torch.tensor(v[2].copy(), dtype=torch.float32, device=a.device)
                    for v in (snapshot.treated, snapshot.matched_control)]
        losses = population_losses(a[None], b[None], observed[0][None], observed[1][None],
                                   raw_scale, model.state_scale, directions)
        rows.append({"target": key.target, "time": key.time,
                     **{k: v.item() for k, v in losses.items()}})
    return {"expression": float(np.mean([r["expression"] for r in rows])),
            "distribution": float(np.mean([r["distribution"] for r in rows])), "rows": rows}


@torch.no_grad()
def integration_check(model, baseline, observations, config):
    """A fixed fitting target at the longest horizon; no response labels used."""
    duration = max(config.fit_times)
    key = min((k for k in observations if k.time == duration), key=lambda k: (k.target, k.dose))
    coarse = predict(model, baseline, key, cells=8, seed=config.seed, step=config.step)
    fine = predict(model, baseline, key, cells=8, seed=config.seed, step=config.step / 2)
    scale = torch.tensor(np.maximum(np.std(baseline, axis=0), 1.), device=model.raw_H.device)
    error = max(((a - b).abs() / scale).max().item() for a, b in zip(coarse, fine))
    return {"target": key.target, "time": duration, "cells": 8,
            "coarse_step": config.step, "fine_step": config.step / 2,
            "max_baseline_scaled_difference": error, "tolerance": .01,
            "passed": math.isfinite(error) and error <= .01,
            "scope": "one fitting-target numerical diagnostic; not propagation or stability proof"}


def _locked_development_scores(model, baseline, data, config, output):
    """Run only after model/graph are saved; no feedback into weight selection."""
    from .q1_pilot_evaluation import (DirectOnlyTransferBaseline, NoResponseBaseline,
                                     aggregate_scores, score_prediction)
    all_data = load_modular_q1(data, scope="all")
    train = set(all_data.provenance["registered_split"]["train"])
    fitting = {k: v for k, v in all_data.observations.items() if k.target in train}
    baselines = {"no_response": NoResponseBaseline.fit(baseline, seed=config.seed),
                 "direct_only_transfer": DirectOnlyTransferBaseline.fit(
                     baseline, fitting, seed=config.seed, times=config.fit_times)}
    scores, arrays = [], {}
    for key, snapshot in sorted(all_data.observations.items(), key=lambda p: (p[0].time, p[0].target)):
        a, b = predict(model, baseline, key, cells=len(snapshot.treated[2]), seed=config.seed, step=config.step)
        predictions = {"graph_rna": (a.cpu().numpy(), b.cpu().numpy()),
                       **{name: method.predict(key) for name, method in baselines.items()}}
        identifier = f"q{key.target}_d{key.dose}_t{key.time}"
        for name, (a, b) in predictions.items():
            for arm, values in (("treated", a), ("control", b)):
                arrays[f"{name}_{identifier}_{arm}"] = values
            for replicate in ((2,) if key.target in train else (0, 1, 2)):
                scores.append({"model": name, "split": "fit_condition_check" if key.target in train else "held_target",
                    "stage": "long_term" if key.time > 8 else "timecourse",
                    **score_prediction(a, b, key, snapshot, replicate=replicate)})
    write_json(output / "scores.json", scores)
    np.savez_compressed(output / "predictions.npz", **arrays)
    return {split: {name: aggregate_scores([r for r in scores if r["model"] == name and r["split"] == split])["mean_metrics"]
            for name in ("graph_rna", *baselines)} for split in ("fit_condition_check", "held_target")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (("updates", 4800), ("cells", 64), ("conditions-per-update", 4),
                          ("condition-microbatch", 1), ("hidden", 8), ("seed", 20261022),
                          ("threads", 4), ("checkpoint-every", 60)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--step", type=float, default=.25)
    parser.add_argument("--lambda-distribution", type=float, default=.1)
    parser.add_argument("--lambda-sparse", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refit", action="store_true", help="fit all 32 train targets AFTER selection")
    parser.add_argument("--probe", action="store_true", help="one longest-time update only; no full training")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.checkpoint_every < 0 or args.probe and args.resume:
        parser.error("invalid threads, checkpoint interval or probe/resume combination")
    if args.output.exists() and not args.resume:
        parser.error("output must be new; previous results are never overwritten")
    if args.resume and (not args.output.is_dir() or (args.output / "report.json").exists()):
        parser.error("resume requires an existing unfinished run")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; activate the Windows Ubuntu GPU environment")
    torch.set_num_threads(args.threads)
    data = load_modular_q1(args.dataset, scope="train")
    baseline = data.baseline[:2].reshape(-1, data.baseline.shape[-1]).copy()
    fitting, validation = (data.observations, {}) if args.refit else internal_split(data.observations, seed=args.seed)
    config = GraphConfig(**{k: getattr(args, k) for k in (
        "updates", "cells", "conditions_per_update", "condition_microbatch", "hidden", "seed",
        "step", "lambda_distribution", "lambda_sparse")},
        fit_times=tuple(sorted({k.time for k in fitting})))
    if args.probe:
        config = replace(config, updates=1, fit_times=(max(config.fit_times),))
        fitting = {k: v for k, v in fitting.items() if k.time in config.fit_times}
    plan = {"schema": "q1-graph-run-v1", "config": asdict(config), "refit": args.refit,
            "probe": args.probe, "provenance": data.provenance, "runtime": runtime(args.device),
            "source_sha256": source_hashes(), "fit_targets": sorted({k.target for k in fitting}),
            "validation_targets": sorted({k.target for k in validation}),
            "held_targets": data.provenance["registered_split"]["held"],
            "fit_replicates": [0, 1], "validation_replicates": [2],
            "state_contract": "mature_RNA_only_effective_Markov_no_hidden_memory",
            "graph_contract": "row_receiver_col_source; Z=W*H elementwise; hard gate threshold .5",
            "loss_contract": "balanced raw-scaled means + lambda_distribution*(marginal+refreshed_joint_log_W2)/2 + lambda_sparse*sum(active_probability)",
            "efficiency": "per_gene_(.5,1); unseen_targets_initial_.75_not_estimated",
            "held_responses_decoded_for_selection": False,
            "test_status": "development_only_previously_inspected_network"}
    plan = json.loads(json.dumps(plan, allow_nan=False))
    if args.resume:
        if json.loads((args.output / "run_plan.json").read_text()) != plan:
            raise ValueError("run registration changed; keep old results and use a new directory")
    else:
        args.output.mkdir(parents=True)
        write_json(args.output / "run_plan.json", plan)
        for name, checksum in plan["source_sha256"].items():
            destination = args.output / "source_snapshot" / name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(Path(__file__).with_name(name).read_bytes())
            if sha256(destination) != checksum:
                raise RuntimeError("source changed during snapshot")
    begin = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def progress(row):
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if row["update"] == 1 or row["update"] % 60 == 0 or row["update"] == config.updates:
            print(json.dumps({k: row[k] for k in ("update", "loss", "expression", "distribution", "expected_edges", "binary_edges_after_update")}), flush=True)

    try:
        model, history = fit_graph(baseline, fitting, config=config, device=args.device,
            checkpoint_path=args.output / "training_checkpoint.pt", checkpoint_every=args.checkpoint_every,
            resume=args.resume, metadata=plan, on_update=progress)
        write_json(args.output / "history.json", history)
        # Reconcile progress after a checkpoint rollback; duplicates are not updates.
        (args.output / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        if args.probe:
            report = {"status": "passed", "schema": "q1-graph-probe-v1", "config": asdict(config),
                "runtime": runtime(args.device), "seconds": time.perf_counter() - begin,
                "scope": "one longest-time forward/backward update; not full training or scientific validation",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                "finite_loss": math.isfinite(history[-1]["loss"]), "update": history[-1]}
            write_json(args.output / "probe.json", report)
            return 0
        torch.save({"schema": "q1-graph-model-v1", "config": asdict(config),
            "n_genes": model.n_genes, "state_dict": model.state_dict()}, args.output / "model.pt")
        graph = {name: value.detach().cpu().numpy() for name, value in model.graph_matrices().items()}
        np.savez_compressed(args.output / "learned_graph.npz", **graph,
                            orientation="row_receiver_col_source", graph_schema="q1-graph-v1")
        for name, matrix in graph.items():
            np.savetxt(args.output / f"{name}.csv", matrix, delimiter=",", fmt="%.9g",
                       header=",".join(f"source_gene_{j}" for j in range(model.n_genes)), comments="")
        used_targets = sorted({q for row in history for q in row["sampled_targets"]})
        write_json(args.output / "efficiency.json", {
            "rho": model.rho.detach().cpu().tolist(), "gradient_training_targets": used_targets,
            "unseen_target_value": .75, "unseen_targets_are_not_estimated": True,
            "note": "training exposure is not proof of identifiable efficiency"})
        lock = {name: sha256(args.output / name) for name in ("model.pt", "learned_graph.npz")}
        write_json(args.output / "prediction_lock.json", lock)
        numerical = integration_check(model, baseline, fitting, config)
        write_json(args.output / "integration_check.json", numerical)
        if not numerical["passed"]:
            raise RuntimeError("integration step-halving diagnostic failed; inspect integration_check.json")
        metrics = validation_metrics(model, baseline, validation, config) if validation else None
        if metrics is not None:
            write_json(args.output / "validation.json", metrics)
        held_scores = (_locked_development_scores(model, baseline, args.dataset, config, args.output)
                       if args.refit else None)
        if any(sha256(args.output / name) != digest for name, digest in lock.items()):
            raise RuntimeError("locked model or graph changed during evaluation")
        report = {**plan, "status": "completed", "seconds": time.perf_counter() - begin,
            "validation_expression": None if metrics is None else metrics["expression"],
            "validation_distribution": None if metrics is None else metrics["distribution"],
            "binary_edges": int(graph["W"].sum()), "expected_edges": float(graph["probability"].sum()),
            "prediction_lock": lock, "integration_check": numerical, "metrics": held_scores,
            "scientific_status": "training_completed; graph_recovery_requires_independent_truth_scoring; no_exact_dynamics_claim"}
        # A visible completion marker must always have a complete checksum list.
        pending = args.output / "report.json.tmp"
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        hashes = {str(p.relative_to(args.output)): sha256(p)
                  for p in sorted(args.output.rglob("*")) if p.is_file()
                  and p.name not in ("report.json", "SHA256SUMS.json") and not p.name.endswith(".tmp")}
        hashes["report.json"] = sha256(pending)
        write_json(args.output / "SHA256SUMS.json", hashes)
        pending.replace(args.output / "report.json")
        return 0
    except Exception as error:
        if not args.resume:
            write_json(args.output / "failure.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
