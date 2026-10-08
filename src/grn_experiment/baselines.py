"""Predeclared prediction comparators fitted from released snapshots only.

Legacy comparators for the self-built sigmoid/Hill fixtures. In particular,
KnownFormRNAODE is not the scmultisim-v1 known-architecture forward model.

The endpoint comparator has no kinetic or GRN interpretation.  The neural ODE
has kinetics but no explicit sparse network.  ``KnownFormRNAODE`` is an
*oracle-form* comparator: it is told which of the two equation families to use,
but receives neither simulator parameters, true edge weights, nor efficiencies.
No class in this module imports or queries the simulator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .learning import TrainConfig, TrainingReport, fit_unpaired_snapshots
from .protocol import Condition, ReplicatedSnapshot


class EndpointResponsePredictor:
    """Fit-only-replicate endpoint effect, held constant at other times.

    For each measured target and dose, the latest *released* sampling time
    supplies a control-adjusted mean effect.  Dose transfer is proportional to
    the nearest observed nominal dose.  For a never-seen target, the prespecified
    fallback is zero response.  Untreated population drift is estimated from
    released matched controls at the nearest observed time.  This intentionally
    simple benchmark cannot recover a GRN or claim a time-resolved mechanism.
    """

    def __init__(self) -> None:
        self._effect: dict[tuple[int, float], np.ndarray] = {}
        self._control: dict[float, np.ndarray] = {}
        self.n_genes: int | None = None

    @property
    def seen_targets(self) -> frozenset[int]:
        return frozenset(target for target, _ in self._effect)

    def fit(self, observations: Mapping[Condition, ReplicatedSnapshot]) -> EndpointResponsePredictor:
        if not observations:
            raise ValueError("at least one released targeted snapshot is required")
        latest: dict[tuple[int, float], tuple[float, np.ndarray]] = {}
        controls: dict[float, list[np.ndarray]] = defaultdict(list)
        n_genes: int | None = None
        for condition, snapshot in observations.items():
            if condition.target is None or condition.time <= 0:
                raise ValueError("only targeted, post-intervention snapshots belong here")
            control = snapshot.fit_control
            if control is None:
                raise ValueError("a matched control is required")
            if n_genes is None:
                n_genes = snapshot.treated.shape[-1]
            elif n_genes != snapshot.treated.shape[-1]:
                raise ValueError("gene dimensions differ between snapshots")
            controls[condition.time].append(control.mean(axis=(0, 1)))
            effect = snapshot.fit_treated.mean(axis=(0, 1)) - control.mean(axis=(0, 1))
            key = (condition.target, condition.dose)
            if key not in latest or condition.time > latest[key][0]:
                latest[key] = (condition.time, effect)
        self._effect = {key: value[1].copy() for key, value in latest.items()}
        self._control = {
            time: np.mean(np.stack(values), axis=0) for time, values in controls.items()
        }
        self.n_genes = n_genes
        return self

    def predict(self, baseline_cells: np.ndarray, condition: Condition) -> np.ndarray:
        cells = np.asarray(baseline_cells, dtype=float)
        if self.n_genes is None:
            raise RuntimeError("fit the endpoint comparator before prediction")
        if cells.ndim != 2 or cells.shape[1] != self.n_genes or not np.isfinite(cells).all():
            raise ValueError("baseline must be a finite cells-by-genes matrix")
        if np.any(cells < 0):
            raise ValueError("baseline RNA must be nonnegative")
        if condition.time == 0:
            return cells.copy()
        nearest_time = min(self._control, key=lambda time: (abs(time - condition.time), time))
        untreated = cells - cells.mean(axis=0, keepdims=True) + self._control[nearest_time]
        if condition.target is None:
            return np.maximum(untreated, 0)
        available = [dose for target, dose in self._effect if target == condition.target]
        if not available:
            return np.maximum(untreated, 0)
        nearest_dose = min(available, key=lambda dose: (abs(dose - condition.dose), dose))
        effect = self._effect[(condition.target, nearest_dose)]
        return np.maximum(untreated + effect * (condition.dose / nearest_dose), 0)


class _PositiveSynthesisODE(nn.Module):
    """Shared numerical solver and sustained target-production attenuation."""

    def __init__(self, n_genes: int, min_efficiency: float) -> None:
        super().__init__()
        if n_genes < 2 or not 0 < min_efficiency < 1:
            raise ValueError("invalid number of genes or efficiency bound")
        self.n_genes = int(n_genes)
        self.min_efficiency = float(min_efficiency)
        self.observed_targets: set[int] = set()
        self.gamma_raw = nn.Parameter(torch.full((n_genes,), 0.5))
        self.rho_raw = nn.Parameter(torch.zeros(n_genes))

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw) + 1e-8

    @property
    def efficiencies(self) -> torch.Tensor:
        return self.min_efficiency + (1 - self.min_efficiency) * torch.sigmoid(self.rho_raw)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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
        production = self.synthesis(torch.clamp(x, min=0.0))
        if target is not None and dose:
            if rho_override is not None and not self.min_efficiency <= rho_override <= 1:
                raise ValueError("efficiency override outside prespecified range")
            efficiency = (
                self.efficiencies[target]
                if rho_override is None
                else torch.as_tensor(rho_override, dtype=x.dtype, device=x.device)
            )
            mask = F.one_hot(torch.tensor(target, device=x.device), self.n_genes).to(x.dtype)
            production = production * (1 - mask * dose * efficiency)
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
        if not np.isfinite(time) or time < 0 or not 0 < step <= 0.25:
            raise ValueError("invalid sampling time or step")
        state = x0
        if time == 0:
            return state
        count = max(1, int(np.ceil(time / step)))
        h = time / count

        def field(value: torch.Tensor) -> torch.Tensor:
            return self.vector_field(value, target, dose, rho_override=rho_override)

        for _ in range(count):
            k1 = field(state)
            k2 = field(state + 0.5 * h * k1)
            k3 = field(state + 0.5 * h * k2)
            k4 = field(state + h * k3)
            state = torch.clamp(state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6, min=0)
        return state

    def edge_penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.gamma_raw.dtype, device=self.gamma_raw.device)


class BlackBoxRNAODE(_PositiveSynthesisODE):
    """Positive-synthesis neural ODE with no explicit interpretable GRN."""

    def __init__(self, n_genes: int = 24, *, hidden: int = 64, min_efficiency: float = 0.5):
        super().__init__(n_genes, min_efficiency)
        if hidden < 2:
            raise ValueError("hidden width must be at least two")
        self.network = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_genes),
        )

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.network(x)) + 1e-8


class KnownFormRNAODE(_PositiveSynthesisODE):
    """Exact simulator equation *form*, with every parameter learned anew.

    The supplied ``family`` is the only oracle information.  Signed ``weights``
    are fitted from snapshots, not copied from simulator truth.  In particular,
    this baseline must never receive the true ``A``, production parameters, or
    target efficiencies during fitting.
    """

    def __init__(
        self,
        family: Literal["sigmoid", "hill"],
        n_genes: int = 24,
        *,
        min_efficiency: float = 0.5,
    ) -> None:
        super().__init__(n_genes, min_efficiency)
        if family not in ("sigmoid", "hill"):
            raise ValueError("family must be sigmoid or hill")
        self.family = family
        self.weights = nn.Parameter(torch.randn(n_genes, n_genes) * 0.03)
        self.basal_raw = nn.Parameter(torch.full((n_genes,), -1.0))
        self.amplitude_raw = nn.Parameter(torch.full((n_genes,), 0.0))
        self.bias = nn.Parameter(torch.zeros(n_genes))
        self.hill_k_raw = nn.Parameter(torch.full((n_genes,), 0.5))
        self.coupling_raw = nn.Parameter(torch.zeros(n_genes))
        self.register_buffer("offdiag", 1 - torch.eye(n_genes))

    @property
    def edge_weights(self) -> torch.Tensor:
        return self.weights * self.offdiag

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.edge_weights
        basal = F.softplus(self.basal_raw) + 1e-8
        amplitude = F.softplus(self.amplitude_raw) + 1e-8
        coupling = torch.sigmoid(self.coupling_raw)
        if self.family == "sigmoid":
            drive = (x - 1.5) @ weights.T
            signal = torch.sigmoid(self.bias + coupling * drive)
            return basal + amplitude * signal
        k = F.softplus(self.hill_k_raw) + 1e-8
        signal = x.square() / (k.square() + x.square())
        positive = F.relu(weights)
        negative = F.relu(-weights)
        drive = signal @ positive.T + (1 - signal) @ negative.T
        return basal + amplitude * (0.5 + coupling * drive)

    def edge_penalty(self) -> torch.Tensor:
        return self.edge_weights.abs().sum() / (self.n_genes * (self.n_genes - 1))


def fit_blackbox_snapshots(
    model: BlackBoxRNAODE,
    baseline: ReplicatedSnapshot,
    observations: Mapping[Condition, ReplicatedSnapshot],
    *,
    config: TrainConfig = TrainConfig(),
) -> TrainingReport:
    """Fit neural dynamics from fit replicates of already released snapshots."""

    return fit_unpaired_snapshots(model, baseline, observations, config=config)


def fit_known_form_snapshots(
    model: KnownFormRNAODE,
    baseline: ReplicatedSnapshot,
    observations: Mapping[Condition, ReplicatedSnapshot],
    *,
    config: TrainConfig = TrainConfig(),
) -> TrainingReport:
    """Oracle-form upper comparator; does not read simulator parameters."""

    return fit_unpaired_snapshots(model, baseline, observations, config=config)
