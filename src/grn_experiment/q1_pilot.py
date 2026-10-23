"""First unknown-form Q1 learner: a development-only latent neural ODE.

Only opened RNA populations and nominal interventions enter this module.
The two latent coordinates per gene are learned proxy coordinates, NOT the
simulator's G/U/M, and their derivatives are not claimed to be a GRN.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import Condition, ReplicatedSnapshot


@dataclass(frozen=True)
class PilotConfig:
    updates: int = 400
    conditions_per_update: int = 4
    cells: int = 64
    projections: int = 24
    hidden: int = 64
    step: float = 0.25
    learning_rate: float = 0.002
    effect_weight: float = 1.0
    target_weight: float = 1.0
    seed: int = 20261022
    fit_times: tuple[float, ...] = (0.25, 4.0)

    def __post_init__(self):
        for name in ("updates", "conditions_per_update", "cells", "projections", "hidden"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.cells < 2 or self.hidden < 2:
            raise ValueError("At least two cells and hidden units are required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("step", "learning_rate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("effect_weight", "target_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.step > 0.25:
            raise ValueError("pilot integration step must not exceed 0.25")
        if (not self.fit_times or any(isinstance(t, bool) or not math.isfinite(t) or t <= 0
                                      for t in self.fit_times)
                or tuple(sorted(set(self.fit_times))) != tuple(self.fit_times)):
            raise ValueError("fit_times must be unique increasing finite positive times")


class LatentPerturbationODE(nn.Module):
    """Generic neural vector field with a shared, gene-local intervention.

    z has two proxy coordinates per gene. Its first coordinate decodes to
    nonnegative RNA through expm1/relu; the second supplies hidden memory.
    There is no prescribed production/decay equation, monotone GRN function,
    true network mask, true efficiency, or per-target categorical embedding.
    The target/dose acts locally and persistently; cross-gene coupling is
    learned in a fully connected neural vector field.
    """

    def __init__(self, n_genes: int, hidden: int = 64):
        super().__init__()
        if n_genes < 2 or hidden < 2:
            raise ValueError("At least two genes and hidden units are required")
        self.n_genes = n_genes
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(n_genes, hidden), nn.Tanh(),
                                     nn.Linear(hidden, n_genes))
        self.core = nn.Sequential(nn.Linear(2 * n_genes, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 2 * n_genes))
        self.local_input = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 2))
        # Small initial drift starts near the measured baseline distribution.
        nn.init.normal_(self.core[-1].weight, std=0.003)
        nn.init.zeros_(self.core[-1].bias)
        nn.init.normal_(self.local_input[-1].weight, std=0.02)
        nn.init.zeros_(self.local_input[-1].bias)

    def initial_state(self, counts: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if counts.shape != noise.shape or counts.shape[-1] != self.n_genes:
            raise ValueError("counts and latent noise must be matching cells-by-genes tensors")
        log_rna = torch.log1p(counts)
        memory = self.encoder(log_rna) + 0.1 * noise
        return torch.stack((log_rna, memory), dim=-1)

    def vector_field(self, state: torch.Tensor, intervention: torch.Tensor) -> torch.Tensor:
        if state.shape[-2:] != (self.n_genes, 2) or intervention.shape != state.shape[:-1]:
            raise ValueError("state/intervention dimensions differ")
        global_update = self.core(state.flatten(-2)).reshape_as(state)
        local_features = torch.cat((state, intervention.unsqueeze(-1)), dim=-1)
        local_update = self.local_input(local_features) * intervention.unsqueeze(-1)
        return global_update + local_update

    @staticmethod
    def decode(state: torch.Tensor) -> torch.Tensor:
        # This is a continuous abundance prediction, not a sample of integer
        # counts or a reconstructed observed single-cell trajectory.
        return torch.expm1(torch.relu(state[..., 0]))

    def integrate(self, initial: torch.Tensor, intervention: torch.Tensor,
                  duration: float, *, step: float = 0.25) -> torch.Tensor:
        if not math.isfinite(duration) or duration < 0 or not math.isfinite(step) or not 0 < step <= 0.25:
            raise ValueError("invalid integration time or step")
        state = initial
        if duration == 0:
            return state
        n_steps = max(1, math.ceil(duration / step))
        h = duration / n_steps
        for _ in range(n_steps):
            k1 = self.vector_field(state, intervention)
            k2 = self.vector_field(state + h * k1 / 2, intervention)
            k3 = self.vector_field(state + h * k2 / 2, intervention)
            k4 = self.vector_field(state + h * k3, intervention)
            state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state


def distribution_loss(predicted: torch.Tensor, observed: torch.Tensor,
                      directions: torch.Tensor) -> torch.Tensor:
    """Squared sliced Wasserstein distance between unpaired log populations."""
    if predicted.shape != observed.shape or predicted.ndim != 3:
        raise ValueError("matching condition/cell/gene minibatches required")
    a = torch.sort(torch.log1p(predicted) @ directions, dim=-2).values
    b = torch.sort(torch.log1p(observed) @ directions, dim=-2).values
    return (a - b).square().mean()


def _observed_batch(chosen, treated, control, rng, cells, device):
    """Independently sample each observed arm; no cross-arm/cross-time pairing."""
    observed_t, observed_c, effects = [], [], []
    for key in chosen:
        a, b = treated[key], control[key]
        left = torch.as_tensor(rng.integers(0, len(a), cells), device=device)
        right = torch.as_tensor(rng.integers(0, len(b), cells), device=device)
        observed_t.append(a[left])
        observed_c.append(b[right])
        effects.append(a.mean(dim=0) - b.mean(dim=0))
    return torch.stack(observed_t), torch.stack(observed_c), torch.stack(effects)


def _training_losses(pred_t, pred_c, observed_t, observed_c, effects,
                     targets, scale, directions):
    error = (pred_t.mean(dim=1) - pred_c.mean(dim=1) - effects) / scale
    return {
        "distribution": distribution_loss(pred_t, observed_t, directions),
        "control": distribution_loss(pred_c, observed_c, directions),
        "effect": error.square().mean(),
        "target_effect": error[torch.arange(len(targets), device=error.device), targets].square().mean(),
    }


def fit_pilot(baseline_fit: np.ndarray,
              observations: dict[Condition, ReplicatedSnapshot], *,
              config: PilotConfig = PilotConfig(), device: str = "cpu",
              on_update: Callable[[dict], None] | None = None) -> tuple[LatentPerturbationODE, list[dict]]:
    """Fixed-budget fitting on explicitly allowed times and fit-only targets.

    The population loss has no cell-row correspondence. Finite minibatch
    resampling need not be identical after reordering the input. Third
    repeats are never read. Predicted arms share initial states only to
    reduce Monte Carlo variance, not to claim observed cells are paired.
    """
    baseline_fit = np.asarray(baseline_fit, dtype=np.float32)
    if (baseline_fit.ndim != 2 or min(baseline_fit.shape) < 2 or
            not np.isfinite(baseline_fit).all() or (baseline_fit < 0).any()):
        raise ValueError("baseline must be finite nonnegative fit-only cells by genes")
    n_genes = baseline_fit.shape[-1]
    if not observations or any(c.target is None or c.time not in config.fit_times for c in observations):
        raise ValueError("pilot fitting accepts only targeted observations at configured fit_times")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = LatentPerturbationODE(n_genes, config.hidden).to(device)
    start = torch.as_tensor(baseline_fit, device=device)
    scale = start.std(dim=0, correction=0).clamp_min(1.0)
    keys = sorted(observations, key=lambda c: (c.time, c.target, c.dose))
    treated, control = {}, {}
    for key in keys:
        snapshot = observations[key]
        if (key.target >= n_genes or snapshot.treated.shape[-1] != n_genes or
                snapshot.fit_control is None):
            raise ValueError("each fit condition needs matching genes and a control")
        treated[key] = torch.as_tensor(snapshot.fit_treated.reshape(-1, n_genes).copy(),
                                       dtype=torch.float32, device=device)
        control[key] = torch.as_tensor(snapshot.fit_control.reshape(-1, n_genes).copy(),
                                       dtype=torch.float32, device=device)
    groups = {t: [k for k in keys if k.time == t] for t in config.fit_times}
    if any(not values for values in groups.values()):
        raise ValueError("all configured fitting times are required")
    if len({tuple((k.target, k.dose) for k in values) for values in groups.values()}) != 1:
        raise ValueError("every fitting time must cover the same target/dose groups")
    directions = F.normalize(torch.randn(n_genes, config.projections, device=device), dim=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    model.train()
    for update in range(config.updates):
        # Equal update frequency per time, including the finite long endpoint.
        duration = config.fit_times[update % len(config.fit_times)]
        pool = groups[duration]
        chosen = [pool[i] for i in rng.choice(len(pool), size=min(len(pool), config.conditions_per_update), replace=False)]
        batch = len(chosen)
        indices = rng.integers(0, len(start), (batch, config.cells))
        initial_counts = start[torch.as_tensor(indices, device=device)]
        noise = torch.randn(initial_counts.shape, device=device)
        initial = model.initial_state(initial_counts, noise)
        intervention = torch.zeros_like(initial_counts)
        for index, key in enumerate(chosen):
            intervention[index, :, key.target] = key.dose
        joint_initial = torch.cat((initial, initial), dim=0)
        joint_input = torch.cat((intervention, torch.zeros_like(intervention)), dim=0)
        prediction = model.decode(model.integrate(joint_initial, joint_input, duration, step=config.step))
        pred_t, pred_c = prediction[:batch], prediction[batch:]
        observed_t, observed_c, effects = _observed_batch(
            chosen, treated, control, rng, config.cells, device)
        target = torch.tensor([key.target for key in chosen], device=device)
        losses = _training_losses(pred_t, pred_c, observed_t, observed_c, effects,
                                  target, scale, directions)
        loss = (losses["distribution"] + losses["control"]
                + config.effect_weight * losses["effect"]
                + config.target_weight * losses["target_effect"])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss; no successful training result may be reported")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"update": update + 1, "time": duration, "loss": float(loss.detach()),
                        **{name: float(value.detach()) for name, value in losses.items()}})
        if on_update is not None:
            on_update(history[-1])
    model.eval()
    return model, history


@torch.no_grad()
def predict_population(model: LatentPerturbationODE, baseline_fit: np.ndarray,
                       condition: Condition, *, cells: int = 128, seed: int = 0,
                       step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Draw predictions from a fit-only baseline; no future observations input."""
    if cells < 2 or (condition.target is not None and condition.target >= model.n_genes):
        raise ValueError("invalid cells or target")
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    baseline_fit = np.asarray(baseline_fit)
    indices = rng.integers(0, len(baseline_fit), cells)
    counts = torch.as_tensor(baseline_fit[indices], dtype=torch.float32, device=device)
    noise = torch.as_tensor(rng.standard_normal(counts.shape), dtype=torch.float32, device=device)
    initial = model.initial_state(counts, noise)
    intervention = torch.zeros_like(counts)
    if condition.target is not None:
        intervention[:, condition.target] = condition.dose
    jointly = model.integrate(torch.stack((initial, initial)),
                              torch.stack((intervention, torch.zeros_like(intervention))),
                              condition.time, step=step)
    predictions = model.decode(jointly).cpu().numpy()
    if not np.isfinite(predictions).all():
        raise RuntimeError("nonfinite RNA predictions")
    return predictions[0], predictions[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_experiment(args):
    """Resolve a frozen development split before any parameter fitting."""
    from .q1_pilot_data import (load_fullcourse_initial, load_modular_q1,
                                load_opened_initial, split_fit_interpolation,
                                split_registered_targets, split_targets)
    if getattr(args, "dataset", None) is not None:
        data = load_modular_q1(args.dataset)
        fitting, held = split_registered_targets(data)
        args.fullcourse = True
        config = PilotConfig(updates=args.updates or 4800, seed=args.seed,
                             learning_rate=0.001, target_weight=0.0,
                             fit_times=tuple(sorted({key.time for key in fitting})))
    elif args.fullcourse:
        data = load_fullcourse_initial(args.archive, args.prefix)
        fitting, held = split_targets(data, seed=args.split_seed)
        config = PilotConfig(updates=args.updates or 600, seed=args.seed,
                             learning_rate=0.001, target_weight=0.0,
                             fit_times=tuple(sorted({key.time for key in fitting})))
    else:
        data = load_opened_initial(args.archive, args.prefix)
        fitting, held = split_fit_interpolation(data)
        config = PilotConfig(updates=args.updates or 400, seed=args.seed)
    return data, fitting, held, config


def _prediction_id(condition: Condition) -> str:
    return f"q{condition.target}_d{condition.dose}_t{condition.time}"


def _evaluate(model, baseline, fitting, held, config, fullcourse):
    """Fit comparators on training data; only then read held-out responses."""
    from .q1_pilot_evaluation import (DirectOnlyTransferBaseline, LinearEffectBaseline,
                                     NoResponseBaseline, score_prediction)
    comparators = {"no_response": NoResponseBaseline.fit(baseline, seed=config.seed)}
    if fullcourse:
        comparators["direct_only_transfer"] = DirectOnlyTransferBaseline.fit(
            baseline, fitting, seed=config.seed, times=config.fit_times)
    else:
        comparators["linear_effect"] = LinearEffectBaseline.fit(baseline, fitting, seed=config.seed)
    scores, predictions = [], {}
    for key, snapshot in sorted({**fitting, **held}.items(), key=lambda item: (item[0].time, item[0].target, item[0].dose)):
        split = ("held_target" if fullcourse else "held_time") if key in held else "fit_condition_check"
        # All held-target replicates are untouched by this fit; fit targets
        # are checked on replicate 2 only. Counts are not network replicates.
        replicates = range(3) if fullcourse and key in held else (2,)
        for name in ("latent_neural_ode", *comparators):
            if name == "latent_neural_ode":
                a, b = predict_population(model, baseline, key, seed=config.seed, step=config.step)
            else:
                a, b = comparators[name].predict(key)
            for replicate in replicates:
                scores.append({"model": name, "split": split,
                               "stage": "long_term" if fullcourse and key.time > 8 else "timecourse",
                               **score_prediction(a, b, key, snapshot, replicate=replicate)})
            predictions[f"{name}_{_prediction_id(key)}_treated"] = a
            predictions[f"{name}_{_prediction_id(key)}_control"] = b
    return scores, predictions


def _summarize_scores(scores, *, group_by):
    from .q1_pilot_evaluation import aggregate_scores
    result = {}
    for group in sorted({row[group_by] for row in scores}, key=str):
        rows = [row for row in scores if row[group_by] == group]
        result[str(group)] = {}
        for name in sorted({row["model"] for row in rows}):
            selected = [row for row in rows if row["model"] == name]
            result[str(group)][name] = aggregate_scores(selected)["mean_metrics"]
    return result


def _integration_check(model, baseline, conditions, predictions, config):
    """Halve RK4 steps over the full horizon; this is not a scientific score."""
    max_cell_error, max_mean_error = 0.0, 0.0
    for key in conditions:
        finer = predict_population(model, baseline, key, seed=config.seed, step=config.step / 2)
        for arm, value in zip(("treated", "control"), finer, strict=True):
            coarse = predictions[f"latent_neural_ode_{_prediction_id(key)}_{arm}"]
            max_cell_error = max(max_cell_error, float(np.max(np.abs(value - coarse))))
            max_mean_error = max(max_mean_error, float(np.max(np.abs(value.mean(0) - coarse.mean(0)))))
    return {"coarse_step": config.step, "fine_step": config.step / 2,
            "conditions": len(conditions), "max_cell_absolute_difference": max_cell_error,
            "max_gene_mean_absolute_difference": max_mean_error,
            "mean_tolerance": 0.01, "cell_tolerance": 0.05,
            "passed": max_mean_error <= 0.01 and max_cell_error <= 0.05,
            "scope": "finite predictions only; not proof of asymptotic stability"}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _snapshot_sources(output: Path, checksums: dict[str, str]) -> None:
    """Archive the registered learner sources before fitting, checking each copy."""
    destination = output / "source_snapshot" / "grn_experiment"
    destination.mkdir(parents=True)
    for name, checksum in checksums.items():
        payload = Path(__file__).with_name(name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise RuntimeError(f"Learner source changed after run registration: {name}")
        copied = destination / name
        copied.write_bytes(payload)
        if _sha256(copied) != checksum:
            raise RuntimeError(f"Learner source snapshot checksum mismatch: {name}")


def _run_experiment(args, data, fitting, held, config):
    baseline = data.baseline[:2].reshape(-1, data.baseline.shape[-1]).copy()
    fit_targets = sorted({key.target for key in fitting})
    test_targets = sorted({key.target for key in held})
    source_hashes = {name: _sha256(Path(__file__).with_name(name)) for name in
                     ("q1_pilot.py", "q1_pilot_data.py", "q1_pilot_evaluation.py", "protocol.py")}
    snapshot_hashes = {**source_hashes, "__init__.py": _sha256(Path(__file__).with_name("__init__.py"))}
    modular = data.provenance.get("schema") == "scmultisim-q1-modular-v1"
    mode = ("modular_fullcourse_target_holdout" if modular else
            "fullcourse_target_holdout" if args.fullcourse else "legacy_time_holdout")
    plan = {"mode": mode,
            "config": asdict(config), "split_seed": None if modular else args.split_seed,
            "split_rule": ("generator public registration fixed from topology before RNA"
                           if modular else "response-independent seeded target split" if args.fullcourse
                           else "registered development time holdout"),
            "fit_targets": fit_targets, "test_targets": test_targets,
            "fit_conditions": len(fitting), "test_conditions": len(held),
            "baseline_replicates": [0, 1], "fit_replicates": [0, 1],
            "test_replicates": [0, 1, 2] if args.fullcourse else [2],
            "checkpoint_selection": "last fixed-budget update; no test-based selection",
            "integration_check_tolerances": {"gene_mean": 0.01, "cell": 0.05},
            "input_provenance": data.provenance, "source_sha256": source_hashes,
            "source_snapshot": {"directory": "source_snapshot/grn_experiment",
                                "files": snapshot_hashes,
                                "scope": "learner and public protocol sources only; not Python or installed dependencies"}}
    _write_json(args.output / "run_plan.json", plan)
    _snapshot_sources(args.output, snapshot_hashes)
    started = time.monotonic()
    with (args.output / "progress.jsonl").open("w") as progress:
        def log_update(row):
            progress.write(json.dumps(row, allow_nan=False) + "\n")
            progress.flush()
            if row["update"] == 1 or row["update"] % 60 == 0:
                print(json.dumps({"update": row["update"], "total": config.updates,
                                  "loss": row["loss"], "seconds": round(time.monotonic() - started, 2)}), flush=True)
        model, history = fit_pilot(baseline, fitting, config=config,
                                   device=args.device, on_update=log_update)
    # Freeze the fitted weights before any held-target scoring.
    torch.save({"state_dict": model.cpu().state_dict(), "n_genes": model.n_genes,
                "hidden": config.hidden, "config": asdict(config),
                "note": "development proxy, not simulator parameters or inferred GRN"}, args.output / "model.pt")
    scores, predictions = _evaluate(model, baseline, fitting, held, config, args.fullcourse)
    check = (_integration_check(model, baseline, tuple({**fitting, **held}), predictions, config)
             if args.fullcourse else {"passed": None, "scope": "not repeated in legacy mode"})
    loss_by_time = {}
    for duration in config.fit_times:
        values = [row["loss"] for row in history if row["time"] == duration]
        loss_by_time[str(duration)] = {"updates": len(values),
                                       "first_10_mean": float(np.mean(values[:10])) if values else None,
                                       "last_10_mean": float(np.mean(values[-10:])) if values else None}
    metrics = _summarize_scores(scores, group_by="split")
    report = {**plan, "status": "completed" if check["passed"] is not False else "numerical_check_failed",
              "scope": "one development network; full-course unseen-target prediction" if args.fullcourse
                       else "one development network; same-target held-time interpolation only",
              "training_device": args.device, "evaluation_device": "cpu", "torch": torch.__version__,
              "numpy": np.__version__, "python": platform.python_version(), "machine": platform.machine(),
              "seconds": time.monotonic() - started, "parameter_count": sum(p.numel() for p in model.parameters()),
              "held_out_responses_and_controls_used_for_fitting": False,
              "loss_by_time": loss_by_time, "metrics": metrics,
              "held_metrics_by_target": _summarize_scores([r for r in scores if r["split"].startswith("held")], group_by="target"),
              "held_metrics_by_stage": _summarize_scores([r for r in scores if r["split"].startswith("held")], group_by="stage"),
              "integration_check": check,
              "limitations": ["pre-registered target holdout on one new development network, not a formal network-blind test"
                               if modular else "previously inspected development data, not a fresh formal blind test",
                              "no GRN or mechanistic recovery claim; non-target is not synonymous with downstream",
                              "RNA population predictions, not paired single-cell trajectories or a count likelihood",
                              "long-term observations use recorded finite durations, not zero-derivative labels",
                              "no confirmed oscillation data or asymptotic model-stability proof",
                              "one network; cell counts and conditions are not independent network samples"]}
    if modular:
        report["limitations"].append("dose 0.7 only; no unseen-dose evaluation in this development run")
    elif args.fullcourse:
        report["limitations"].append("dose 0.7 only; dose 0.3 lacks long-term observations")
    _write_json(args.output / "history.json", history)
    _write_json(args.output / "scores.json", scores)
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    _write_json(args.output / "report.json", report)
    _write_json(args.output / "SHA256SUMS.json", {
        p.name: _sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "seconds": report["seconds"], "metrics": metrics,
                      "integration_check": check}, indent=2), flush=True)
    return 0 if report["status"] == "completed" else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="legacy development tar archive")
    source.add_argument("--dataset", type=Path, help="local public modular-Q1 dataset directory")
    parser.add_argument("--prefix", default="grn-scmultisim-20261014-dev1/outputs/scmultisim-dev-101/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--fullcourse", action="store_true", help="dose 0.7, all available times, 4/2 target split")
    parser.add_argument("--split-seed", type=int, default=20261022)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be a new directory; previous runs are never overwritten")
    if args.threads < 1 or args.split_seed < 0 or (args.updates is not None and args.updates < 1):
        parser.error("threads/updates must be positive and split-seed nonnegative")
    torch.set_num_threads(args.threads)
    data, fitting, held, config = _prepare_experiment(args)
    args.output.mkdir(parents=True)
    try:
        return _run_experiment(args, data, fitting, held, config)
    except Exception as error:
        _write_json(args.output / "failure.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
