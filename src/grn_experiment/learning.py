"""Interpretable RNA ODE learner trained only on unpaired snapshots.

Importing this module never imports the simulator.  True vector fields and
Jacobians are accepted only by the isolated development audit function.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import COMMON_BASELINE, Condition, ReplicatedSnapshot


class SparseRNAODE(nn.Module):
    """Unknown monotone synthesis functions with a sparse signed edge matrix.

    The broad additive-input/monotone-link envelope is not given either exact
    generator family.  `weights[i,j]` means regulator j affects target i.
    Positive transforms and links make its sign agree with the local
    off-diagonal derivative, but its magnitude is not raw simulator A.
    """

    def __init__(
        self,
        n_genes: int = 24,
        *,
        min_efficiency: float = 0.5,
        allowed_edges: np.ndarray | None = None,
    ):
        super().__init__()
        if n_genes < 2 or not 0 < min_efficiency < 1:
            raise ValueError("invalid number of genes or efficiency bound")
        self.n_genes = n_genes
        self.min_efficiency = float(min_efficiency)
        self.weights = nn.Parameter(torch.randn(n_genes, n_genes) * 0.03)
        self.slopes_raw = nn.Parameter(torch.zeros(n_genes, 6))
        self.link_raw = nn.Parameter(torch.full((n_genes, 9), -2.0))
        self.basal_raw = nn.Parameter(torch.full((n_genes,), -1.0))
        self.gamma_raw = nn.Parameter(torch.full((n_genes,), 0.5))
        self.bias = nn.Parameter(torch.zeros(n_genes))
        self.rho_raw = nn.Parameter(torch.zeros(n_genes))
        self.register_buffer("knots", torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]))
        self.register_buffer("link_centers", torch.linspace(-4.0, 4.0, 9))
        self.register_buffer("offdiag", 1.0 - torch.eye(n_genes))
        if allowed_edges is None:
            allowed = np.ones((n_genes, n_genes), dtype=bool)
        else:
            allowed = np.asarray(allowed_edges, dtype=bool)
            if allowed.shape != (n_genes, n_genes):
                raise ValueError("allowed_edges must be a square genes-by-genes mask")
        self.register_buffer("allowed_edges", torch.as_tensor(allowed.copy(), dtype=torch.float32))
        # This set records which target efficiencies have actually been
        # informed by an acquired response, not ground-truth efficiencies.
        self.observed_targets: set[int] = set()

    @property
    def edge_weights(self) -> torch.Tensor:
        return self.weights * self.offdiag * self.allowed_edges

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw) + 0.01

    @property
    def efficiencies(self) -> torch.Tensor:
        return self.min_efficiency + (1.0 - self.min_efficiency) * torch.sigmoid(self.rho_raw)

    def _monotone_regulators(self, x: torch.Tensor) -> torch.Tensor:
        state = torch.clamp(x, min=0.0)
        segments = torch.clamp(
            state[..., :, None] - self.knots[:-1],
            min=0.0,
        )
        widths = self.knots[1:] - self.knots[:-1]
        segments = torch.minimum(segments, widths)
        slopes = F.softplus(self.slopes_raw) + 0.01
        transformed = (segments * slopes).sum(dim=-1)
        transformed = transformed + F.relu(state - self.knots[-1]) * slopes[:, -1]
        unit_value = (widths[:2] * slopes[:, :2]).sum(dim=-1)
        return transformed / unit_value.clamp_min(1e-8)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        drive = self._monotone_regulators(x) @ self.edge_weights.T + self.bias
        basis = torch.sigmoid(drive[..., :, None] - self.link_centers)
        return F.softplus(self.basal_raw) + (basis * F.softplus(self.link_raw)).sum(-1)

    def vector_field(
        self,
        x: torch.Tensor,
        target: int | None = None,
        dose: float = 0.0,
        *,
        rho_override: float | None = None,
    ) -> torch.Tensor:
        if x.shape[-1] != self.n_genes or not 0 <= dose <= 1:
            raise ValueError("invalid state or dose")
        if target is None and dose != 0:
            raise ValueError("nonzero dose requires a target")
        if target is not None and not 0 <= target < self.n_genes:
            raise ValueError("invalid target")
        production = self.synthesis(x)
        if target is not None and dose:
            efficiency = (
                self.efficiencies[target]
                if rho_override is None
                else torch.as_tensor(rho_override, dtype=x.dtype, device=x.device)
            )
            if rho_override is not None and not self.min_efficiency <= rho_override <= 1:
                raise ValueError("efficiency override outside prespecified range")
            attenuation = 1.0 - dose * efficiency
            mask = F.one_hot(
                torch.tensor(target, device=x.device), num_classes=self.n_genes
            ).to(dtype=x.dtype)
            production = production * (1.0 - mask + mask * attenuation)
        return production - self.gamma * x

    def integrate(
        self,
        x0: torch.Tensor,
        time: float,
        target: int | None = None,
        dose: float = 0.0,
        *,
        step: float = 0.1,
        rho_override: float | None = None,
    ) -> torch.Tensor:
        """Differentiable RK4; `time=0` is always pre-intervention."""

        if time < 0 or not np.isfinite(time) or not 0 < step <= 0.25:
            raise ValueError("invalid sampling time or integration step")
        state = x0
        if time == 0:
            return state
        count = max(1, int(np.ceil(time / step)))
        h = time / count

        def field(value: torch.Tensor) -> torch.Tensor:
            return self.vector_field(
                torch.clamp(value, min=0.0), target, dose, rho_override=rho_override
            )

        for _ in range(count):
            k1 = field(state)
            k2 = field(state + 0.5 * h * k1)
            k3 = field(state + 0.5 * h * k2)
            k4 = field(state + h * k3)
            state = torch.clamp(state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, min=0.0)
        return state

    def local_jacobian(self, state: torch.Tensor) -> torch.Tensor:
        """Full state Jacobian at one state; development/scoring use only."""

        if state.ndim != 1 or state.numel() != self.n_genes:
            raise ValueError("one RNA state is required")
        return torch.autograd.functional.jacobian(
            lambda value: self.vector_field(value), state
        )

    def edge_penalty(self) -> torch.Tensor:
        return self.edge_weights.abs().sum() / (self.n_genes * (self.n_genes - 1))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 250
    conditions_per_epoch: int = 4
    cells_per_condition: int = 128
    projections: int = 32
    learning_rate: float = 0.001
    sparsity_weight: float = 0.001
    effect_weight: float = 0.2
    step: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (
                self.epochs,
                self.conditions_per_epoch,
                self.cells_per_condition,
                self.projections,
            )
        ):
            raise ValueError("training counts must be positive integers")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not np.isfinite(self.sparsity_weight)
            or self.sparsity_weight < 0
            or not np.isfinite(self.effect_weight)
            or self.effect_weight < 0
            or not np.isfinite(self.step)
            or not 0 < self.step <= 0.25
        ):
            raise ValueError("invalid training rate, weights, or integration step")


@dataclass(frozen=True)
class TrainingReport:
    losses: tuple[float, ...]
    conditions_seen: int
    cells_used_per_condition: int


def torch_sliced_wasserstein(
    predicted: torch.Tensor, observed: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    if predicted.ndim != 2 or observed.ndim != 2:
        raise ValueError("cells-by-genes tensors required")
    if predicted.shape != observed.shape or predicted.shape[1] != directions.shape[0]:
        raise ValueError("equal minibatch shapes and matching projections required")
    a = torch.sort(predicted @ directions, dim=0).values
    b = torch.sort(observed @ directions, dim=0).values
    return torch.sqrt(torch.mean((a - b) ** 2) + 1e-12)


def fit_unpaired_snapshots(
    model: SparseRNAODE,
    baseline: ReplicatedSnapshot,
    observations: Mapping[Condition, ReplicatedSnapshot],
    *,
    config: TrainConfig = TrainConfig(),
) -> TrainingReport:
    """Fit only on two declared fit replicates, never pair cells across time.

    Callers must pass *only* observations already released by the protocol
    gate.  The third replicate, future candidate outcomes, validation and
    final test observations are not read here.
    """

    if not observations or config.epochs < 1 or config.cells_per_condition < 2:
        raise ValueError("nonempty observations and positive training budget required")
    if any(c == COMMON_BASELINE or c.time <= 0 for c in observations):
        raise ValueError("baseline is supplied separately and cannot be an outcome")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    start = torch.as_tensor(
        baseline.fit_treated.reshape(-1, model.n_genes), dtype=dtype, device=device
    )
    conditions = tuple(sorted(observations, key=lambda c: (c.target is None, c.target or -1, c.dose, c.time)))
    projection = torch.randn(model.n_genes, config.projections, device=device, dtype=dtype)
    projection = F.normalize(projection, dim=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        batch_indices = rng.choice(
            len(conditions), size=min(config.conditions_per_epoch, len(conditions)), replace=False
        )
        optimizer.zero_grad()
        total = model.edge_penalty() * config.sparsity_weight
        for index in batch_indices:
            condition = conditions[int(index)]
            sample = observations[condition]
            if sample.fit_control is None:
                raise ValueError("each treated snapshot needs a matched control")
            count = config.cells_per_condition
            baseline_rows = rng.choice(len(start), size=count, replace=len(start) < count)
            treated_pool = sample.fit_treated.reshape(-1, model.n_genes)
            control_pool = sample.fit_control.reshape(-1, model.n_genes)
            treated_rows = rng.choice(len(treated_pool), size=count, replace=len(treated_pool) < count)
            control_rows = rng.choice(len(control_pool), size=count, replace=len(control_pool) < count)
            x0 = start[baseline_rows]
            observed_treated = torch.as_tensor(
                treated_pool[treated_rows],
                device=device,
                dtype=dtype,
            )
            observed_control = torch.as_tensor(
                control_pool[control_rows],
                device=device,
                dtype=dtype,
            )
            predicted_treated = model.integrate(
                x0, condition.time, condition.target, condition.dose, step=config.step
            )
            predicted_control = model.integrate(x0, condition.time, step=config.step)
            distance = torch_sliced_wasserstein(predicted_treated, observed_treated, projection)
            distance = distance + torch_sliced_wasserstein(
                predicted_control, observed_control, projection
            )
            effect_error = torch.mean(
                (
                    predicted_treated.mean(0)
                    - predicted_control.mean(0)
                    - observed_treated.mean(0)
                    + observed_control.mean(0)
                ) ** 2
            )
            total = total + (distance + config.effect_weight * effect_error) / len(batch_indices)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    model.observed_targets.update(
        int(condition.target) for condition in conditions if condition.target is not None
    )
    return TrainingReport(tuple(losses), len(conditions), config.cells_per_condition)


@torch.no_grad()
def predict_population(
    model: SparseRNAODE,
    baseline_cells: np.ndarray,
    condition: Condition,
    *,
    rho_override: float | None = None,
    step: float = 0.1,
) -> np.ndarray:
    """Predict without touching withheld observations or true efficiency.

    For an unobserved target and no override, the nominal point prediction
    uses the prior midpoint (0.75 for the default [0.5, 1] range).  It is not
    an efficiency estimate; report a range via ``predict_efficiency_interval``.
    """

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    initial = torch.as_tensor(baseline_cells, device=device, dtype=dtype)
    if condition.target is not None and condition.target not in model.observed_targets and rho_override is None:
        rho_override = (model.min_efficiency + 1.0) / 2.0
    return (
        model.integrate(
            initial,
            condition.time,
            condition.target,
            condition.dose,
            step=step,
            rho_override=rho_override,
        )
        .cpu()
        .numpy()
    )


@dataclass(frozen=True)
class EfficiencyInterval:
    """A sensitivity envelope, not a calibrated confidence interval."""

    lower: np.ndarray
    upper: np.ndarray
    efficiency_values: tuple[float, ...]


def predict_efficiency_interval(
    model: SparseRNAODE,
    baseline_cells: np.ndarray,
    condition: Condition,
    *,
    efficiency_values: tuple[float, ...] = (0.5, 0.625, 0.75, 0.875, 1.0),
    step: float = 0.1,
) -> EfficiencyInterval:
    """Profile an unseen target's fixed-but-unknown efficiency range.

    The same efficiency is used for that target at every dose and time in any
    one prediction.  This envelope isolates efficiency sensitivity only;
    model and sampling uncertainty require a separately calibrated ensemble.
    """

    if not efficiency_values or any(
        not np.isfinite(value) or not model.min_efficiency <= value <= 1
        for value in efficiency_values
    ):
        raise ValueError("efficiency values must lie in the prespecified range")
    outcomes = np.stack(
        [
            predict_population(
                model, baseline_cells, condition, rho_override=value, step=step
            )
            for value in efficiency_values
        ]
    )
    return EfficiencyInterval(
        outcomes.min(axis=0), outcomes.max(axis=0), tuple(efficiency_values)
    )


@dataclass(frozen=True)
class CapacityReport:
    relative_vector_field_rmse: float
    strong_edge_sign_accuracy: float
    passed: bool


def audit_function_family(
    model: SparseRNAODE,
    states: np.ndarray,
    true_field: Callable[[np.ndarray], np.ndarray],
    true_jacobian: Callable[[np.ndarray], np.ndarray],
    *,
    epochs: int = 500,
    learning_rate: float = 0.003,
    strong_quantile: float = 0.75,
) -> CapacityReport:
    """Development-only oracle-capacity audit, never used for snapshot fitting.

    Fits vector-field labels at simulator-held states in a private clone,
    then checks local Jacobian signs. The passed model is never modified,
    avoiding oracle-parameter leakage into the main snapshot learner. This
    is an expressivity/optimization gate, *not* a demonstration that
    unpaired snapshots identify the field.
    """

    points = np.asarray(states, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != model.n_genes or epochs < 1:
        raise ValueError("invalid development state grid")
    model = deepcopy(model)
    device = next(model.parameters()).device
    x = torch.as_tensor(points, device=device)
    truth = torch.as_tensor(np.asarray(true_field(points)), device=device, dtype=x.dtype)
    if truth.shape != x.shape:
        raise ValueError("true_field must return one vector per state")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        error = torch.mean((model.vector_field(x) - truth) ** 2)
        error.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = model.vector_field(x)
        relative = float(
            (torch.mean((predicted - truth) ** 2).sqrt()
            / torch.mean(truth**2).sqrt().clamp_min(1e-6)).cpu()
        )
    predicted_j = np.stack(
        [model.local_jacobian(x[index].detach()).detach().cpu().numpy() for index in range(len(x))]
    )
    true_j = np.stack([np.asarray(true_jacobian(point)) for point in points])
    if true_j.shape != predicted_j.shape:
        raise ValueError("true_jacobian returned an invalid shape")
    mask = ~np.eye(model.n_genes, dtype=bool)
    magnitudes = np.abs(true_j[..., mask])
    cutoff = np.quantile(magnitudes, strong_quantile)
    strong = magnitudes > max(cutoff, 1e-10)
    sign_accuracy = (
        float(np.mean(np.sign(predicted_j[..., mask][strong]) == np.sign(true_j[..., mask][strong])))
        if np.any(strong)
        else float("nan")
    )
    return CapacityReport(relative, sign_accuracy, bool(relative <= 0.05 and sign_accuracy >= 0.95))
