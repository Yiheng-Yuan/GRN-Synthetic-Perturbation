"""Shared fitting/evaluation for development-only Q1 neural dynamics.

Only opened RNA populations and nominal interventions enter this module.
The original latent arm uses two proxy coordinates per gene. The optional
RNA-current/history arms retain only RNA, with an explicit delayed RNA buffer
in the history arm. No model's derivatives are claimed to be a recovered GRN.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
    condition_microbatch: int | None = None
    model_kind: str = "latent_neural_ode"
    history_lag: float = 1.0

    def __post_init__(self):
        if self.model_kind not in ("latent_neural_ode", "rna_current", "rna_history"):
            raise ValueError("unknown model_kind")
        if (isinstance(self.history_lag, bool) or not math.isfinite(self.history_lag)
                or self.history_lag < self.step):
            raise ValueError("history_lag must be finite and at least the integration step")
        for name in ("updates", "conditions_per_update", "cells", "projections", "hidden"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.cells < 2 or self.hidden < 2:
            raise ValueError("At least two cells and hidden units are required")
        if (self.condition_microbatch is not None and
                (isinstance(self.condition_microbatch, bool) or
                 not isinstance(self.condition_microbatch, int) or self.condition_microbatch < 1)):
            raise ValueError("condition_microbatch must be a positive integer or None")
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


def build_model(n_genes: int, config: PilotConfig) -> nn.Module:
    """Construct the registered state space; RNA arms never create latent h."""
    if config.model_kind == "latent_neural_ode":
        return LatentPerturbationODE(n_genes, config.hidden)
    from .q1_rna import RNAPerturbationODE
    return RNAPerturbationODE(n_genes, config.hidden,
        history_lag=config.history_lag if config.model_kind == "rna_history" else None)


def _learner_sources(config: PilotConfig) -> tuple[str, ...]:
    names = ("q1_pilot.py", "q1_pilot_data.py", "q1_pilot_evaluation.py", "protocol.py")
    return names + (("q1_rna.py",) if config.model_kind != "latent_neural_ode" else ())


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


def _checkpoint_context(baseline, observations, config, device, metadata):
    """Fingerprint only fit inputs; the third repeat never enters fitting."""
    digest = hashlib.sha256()
    array = np.ascontiguousarray(baseline, dtype=np.float32)
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    for key in sorted(observations, key=lambda c: (c.time, c.target, c.dose)):
        digest.update(json.dumps([key.target, key.dose, key.time]).encode())
        snapshot = observations[key]
        for values in (snapshot.fit_treated, snapshot.fit_control):
            array = np.ascontiguousarray(values, dtype=np.float32)
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    runtime = {"torch": str(torch.__version__), "numpy": np.__version__,
               "python": platform.python_version(), "machine": platform.machine()}
    if torch.device(device).type == "cuda":
        runtime.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(device),
                       capability=torch.cuda.get_device_capability(device))
    # Normalize tuples and JSON primitives identically in fresh/resumed runs.
    return json.loads(json.dumps({"config": asdict(config), "device": str(device),
                                 "fit_input_sha256": digest.hexdigest(),
                                 "learner_sha256": _sha256(Path(__file__)),
                                 "model_sources_sha256": {
                                     name: _sha256(Path(__file__).with_name(name))
                                     for name in _learner_sources(config)},
                                 "runtime": runtime,
                                 "metadata": metadata}, allow_nan=False))


def _load_training_checkpoint(path, context, updates):
    """Only load trusted local checkpoints; refuse incompatible continuations."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "q1-training-v1" or payload.get("context") != context:
        raise ValueError("training checkpoint is incompatible with config, data, source or device")
    completed, history = payload.get("completed_updates"), payload.get("history")
    if (isinstance(completed, bool) or not isinstance(completed, int) or
            not 0 <= completed <= updates or not isinstance(history, list) or len(history) != completed):
        raise ValueError("training checkpoint has invalid completed-update history")
    times = context["config"]["fit_times"]
    for index, row in enumerate(history):
        if (row.get("update") != index + 1 or row.get("time") != times[index % len(times)] or
                any(not isinstance(row.get(name), (float, int)) or not math.isfinite(row[name])
                    for name in ("loss", "distribution", "control", "effect", "target_effect"))):
            raise ValueError("training checkpoint has invalid completed-update history")
    return payload


def _save_training_checkpoint(path, context, model, optimizer, rng, directions, history, device):
    """Atomic replacement leaves the last completed update usable after failure."""
    payload = {"schema": "q1-training-v1", "context": context,
               "completed_updates": len(history), "history": history,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else [],
               "directions": directions.detach().cpu()}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_pilot(baseline_fit: np.ndarray,
              observations: dict[Condition, ReplicatedSnapshot], *,
              config: PilotConfig = PilotConfig(), device: str = "cpu",
              on_update: Callable[[dict], None] | None = None,
              checkpoint_path: Path | None = None, checkpoint_every: int = 0,
              resume: bool = False, checkpoint_metadata: dict | None = None
              ) -> tuple[nn.Module, list[dict]]:
    """Fixed-budget fitting on explicitly allowed times and fit-only targets.

    The population loss has no cell-row correspondence. Finite minibatch
    resampling need not be identical after reordering the input. Third
    repeats are never read. Predicted arms share initial states only to
    reduce Monte Carlo variance, not to claim observed cells are paired.
    Condition microbatches accumulate their condition-weighted losses before
    one clipping/Adam step; cell counts and the effective update stay fixed.
    """
    baseline_fit = np.asarray(baseline_fit, dtype=np.float32)
    if (baseline_fit.ndim != 2 or min(baseline_fit.shape) < 2 or
            not np.isfinite(baseline_fit).all() or (baseline_fit < 0).any()):
        raise ValueError("baseline must be finite nonnegative fit-only cells by genes")
    n_genes = baseline_fit.shape[-1]
    if not observations or any(c.target is None or c.time not in config.fit_times for c in observations):
        raise ValueError("pilot fitting accepts only targeted observations at configured fit_times")
    if (isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 0 or
            (checkpoint_every and checkpoint_path is None) or (resume and checkpoint_path is None)):
        raise ValueError("checkpoint_every must be nonnegative and checkpoint/resume needs checkpoint_path")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None and not checkpoint_path.parent.is_dir():
        raise ValueError("checkpoint parent must already exist")
    if checkpoint_path is not None and checkpoint_path.exists() and not resume:
        raise ValueError("existing training checkpoint requires explicit resume")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = build_model(n_genes, config).to(device)
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
    if config.model_kind == "latent_neural_ode":
        directions = F.normalize(torch.randn(n_genes, config.projections, device=device), dim=0)
    else:
        # Architecture widths consume different initialization draws. Use an
        # independent CPU stream so both RNA arms see exactly the same loss.
        projection_rng = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        directions = F.normalize(torch.randn(n_genes, config.projections,
                                             generator=projection_rng), dim=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    context = (_checkpoint_context(baseline_fit, observations, config, device, checkpoint_metadata)
               if checkpoint_path is not None else None)
    if resume:
        saved = _load_training_checkpoint(checkpoint_path, context, config.updates)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        directions = saved["directions"].to(device)
        history = saved["history"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.device(device).type == "cuda":
            if len(saved["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("training checkpoint CUDA device count changed")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    model.train()
    for update in range(len(history), config.updates):
        # Equal update frequency per time, including the finite long endpoint.
        duration = config.fit_times[update % len(config.fit_times)]
        pool = groups[duration]
        chosen = [pool[i] for i in rng.choice(len(pool), size=min(len(pool), config.conditions_per_update), replace=False)]
        batch = len(chosen)
        indices = rng.integers(0, len(start), (batch, config.cells))
        initial_counts = start[torch.as_tensor(indices, device=device)]
        noise = (torch.randn(initial_counts.shape, device=device)
                 if config.model_kind == "latent_neural_ode" else None)
        intervention = torch.zeros_like(initial_counts)
        for index, key in enumerate(chosen):
            intervention[index, :, key.target] = key.dose
        observed_t, observed_c, effects = _observed_batch(
            chosen, treated, control, rng, config.cells, device)
        target = torch.tensor([key.target for key in chosen], device=device)
        optimizer.zero_grad(set_to_none=True)
        totals = {name: 0.0 for name in ("loss", "distribution", "control", "effect", "target_effect")}
        microbatch = config.condition_microbatch or batch
        for offset in range(0, batch, microbatch):
            end = min(batch, offset + microbatch)
            part = slice(offset, end)
            count, weight = end - offset, (end - offset) / batch
            # Re-encode per chunk: sharing an encoder graph across backward()
            # calls would retain the long integration graph or reuse it unsafely.
            initial = (model.initial_state(initial_counts[part], noise[part])
                       if noise is not None else model.initial_state(initial_counts[part]))
            joint_initial = torch.cat((initial, initial), dim=0)
            inputs = intervention[part]
            joint_input = torch.cat((inputs, torch.zeros_like(inputs)), dim=0)
            prediction = model.decode(model.integrate(joint_initial, joint_input, duration, step=config.step))
            losses = _training_losses(prediction[:count], prediction[count:], observed_t[part],
                                      observed_c[part], effects[part], target[part], scale, directions)
            loss = (losses["distribution"] + losses["control"]
                    + config.effect_weight * losses["effect"]
                    + config.target_weight * losses["target_effect"])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss; no successful training result may be reported")
            (weight * loss).backward()
            totals["loss"] += weight * float(loss.detach())
            for name, value in losses.items():
                totals[name] += weight * float(value.detach())
            del prediction, losses, loss, initial, joint_initial, joint_input
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"update": update + 1, "time": duration, **totals})
        if checkpoint_path is not None and (update + 1 == config.updates or
                checkpoint_every and (update + 1) % checkpoint_every == 0):
            _save_training_checkpoint(checkpoint_path, context, model, optimizer, rng, directions, history, device)
        if on_update is not None:
            on_update(history[-1])
    model.eval()
    return model, history


@torch.no_grad()
def predict_population(model: nn.Module, baseline_fit: np.ndarray,
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
    if isinstance(model, LatentPerturbationODE):
        noise = torch.as_tensor(rng.standard_normal(counts.shape), dtype=torch.float32, device=device)
        initial = model.initial_state(counts, noise)
    else:
        initial = model.initial_state(counts)
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
    config = replace(config, condition_microbatch=getattr(args, "condition_microbatch", None),
                     model_kind=getattr(args, "model_kind", "latent_neural_ode"),
                     history_lag=getattr(args, "history_lag", 1.0),
                     hidden=config.hidden if getattr(args, "hidden", None) is None else args.hidden)
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
        for name in (config.model_kind, *comparators):
            if name == config.model_kind:
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
            coarse = predictions[f"{config.model_kind}_{_prediction_id(key)}_{arm}"]
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
    source_hashes = {name: _sha256(Path(__file__).with_name(name))
                     for name in _learner_sources(config)}
    snapshot_hashes = {**source_hashes, "__init__.py": _sha256(Path(__file__).with_name("__init__.py"))}
    modular = data.provenance.get("schema") == "scmultisim-q1-modular-v1"
    mode = ("modular_fullcourse_target_holdout" if modular else
            "fullcourse_target_holdout" if args.fullcourse else "legacy_time_holdout")
    plan = {"mode": mode, "model_kind": config.model_kind,
            "state_contract": ({"state": "log-RNA plus learned latent coordinate per gene",
                                 "integrator": "RK4"} if config.model_kind == "latent_neural_ode" else
                                {"state": "nonnegative RNA only; identity decoder",
                                 "integrator": ("projected RK4; linear interpolation of generated history"
                                                if config.model_kind == "rna_history" else "projected RK4"),
                                 "history_lag": config.history_lag if config.model_kind == "rna_history" else None,
                                 "pre_zero_history": "constant initial RNA" if config.model_kind == "rna_history" else None,
                                 "observed_history_inputs": False, "latent_noise": False}),
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
    checkpoint_every = getattr(args, "checkpoint_every", 0)
    resume = getattr(args, "resume", False)
    checkpoint_path = args.output / "training_checkpoint.pt" if checkpoint_every or resume else None
    plan["training_checkpoint"] = {"file": "training_checkpoint.pt" if checkpoint_path else None,
                                   "every_updates": checkpoint_every,
                                   "scope": "completed optimizer updates; resume requires identical registration"}
    # Round-trip tuples just as the saved plan does, before equality checks.
    plan = json.loads(json.dumps(plan, allow_nan=False))
    previous_history = []
    if resume:
        if json.loads((args.output / "run_plan.json").read_text()) != plan:
            raise ValueError("resume run_plan differs from current config, data provenance or source")
        for name, checksum in snapshot_hashes.items():
            if _sha256(args.output / "source_snapshot" / "grn_experiment" / name) != checksum:
                raise ValueError(f"resume source snapshot mismatch: {name}")
        context = _checkpoint_context(baseline, fitting, config, args.device, plan)
        previous_history = _load_training_checkpoint(checkpoint_path, context, config.updates)["history"]
    else:
        _write_json(args.output / "run_plan.json", plan)
        _snapshot_sources(args.output, snapshot_hashes)
    started = time.monotonic()
    progress = None

    def open_progress():
        # A failed update after the last checkpoint is replayed, never counted
        # twice. Regenerate the log from the validated checkpoint history.
        stream = (args.output / "progress.jsonl").open("w")
        for row in previous_history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return stream

    def log_update(row):
        nonlocal progress
        # Delay writes until checkpoint model/optimizer/RNG restoration and
        # a complete update succeed. An invalid resume leaves files untouched.
        if progress is None:
            progress = open_progress()
        progress.write(json.dumps(row, allow_nan=False) + "\n")
        progress.flush()
        if row["update"] == 1 or row["update"] % 60 == 0:
            print(json.dumps({"update": row["update"], "total": config.updates,
                              "loss": row["loss"], "seconds": round(time.monotonic() - started, 2)}), flush=True)
    try:
        model, history = fit_pilot(baseline, fitting, config=config,
                                   device=args.device, on_update=log_update,
                                   checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every,
                                   resume=resume, checkpoint_metadata=plan)
        if progress is None:
            progress = open_progress()  # All updates were already checkpointed.
    finally:
        if progress is not None:
            progress.close()
    if resume and (args.output / "failure.json").is_file():
        (args.output / "failure.json").rename(args.output / f"failure-before-resume-{time.time_ns()}.json")
    # Freeze the fitted weights before any held-target scoring.
    torch.save({"state_dict": model.cpu().state_dict(), "n_genes": model.n_genes,
                "hidden": config.hidden, "config": asdict(config), "model_kind": config.model_kind,
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
              "resumed_from_update": len(previous_history) if resume else None,
              "seconds_scope": "this process only; excludes any prior interrupted attempts",
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
        if config.model_kind != "latent_neural_ode":
            report["limitations"].extend([
                "held-target results were inspected in the earlier latent-model run; this is a development comparison",
                "shared original population/effect loss retained; improved accuracy is not guaranteed",
                "positivity projection and interpolated delay require the reported step-halving check",
                "unpaired snapshots do not establish true cell histories or biological Markov order"])
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
    parser.add_argument("--model-kind", choices=("latent_neural_ode", "rna_current", "rna_history"),
                        default="latent_neural_ode")
    parser.add_argument("--history-lag", type=float, default=1.0,
                        help="fixed delay in simulation time units, not a count of sampling times")
    parser.add_argument("--hidden", type=int, help="MLP width; temporary features, not persistent memory")
    parser.add_argument("--fullcourse", action="store_true", help="dose 0.7, all available times, 4/2 target split")
    parser.add_argument("--split-seed", type=int, default=20261022)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--condition-microbatch", type=int,
                        help="conditions per backward pass; effective update size and cells are unchanged")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="save resumable optimizer/RNG state every N updates (0 disables checkpoints)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the same registered output from its last completed-update checkpoint")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.resume:
        parser.error("output must be a new directory; previous runs are never overwritten")
    if args.resume and (not args.output.is_dir() or (args.output / "report.json").exists()):
        parser.error("resume requires an existing unfinished run directory")
    if args.threads < 1 or args.split_seed < 0 or (args.updates is not None and args.updates < 1):
        parser.error("threads/updates must be positive and split-seed nonnegative")
    if args.checkpoint_every < 0 or (args.condition_microbatch is not None and args.condition_microbatch < 1):
        parser.error("checkpoint-every must be nonnegative and condition-microbatch positive")
    torch.set_num_threads(args.threads)
    data, fitting, held, config = _prepare_experiment(args)
    if not args.resume:
        args.output.mkdir(parents=True)
    try:
        return _run_experiment(args, data, fitting, held, config)
    except Exception as error:
        # Incompatible resumes are read-only: preserve all prior evidence.
        if not args.resume:
            _write_json(args.output / "failure.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
