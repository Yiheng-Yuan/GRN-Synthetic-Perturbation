"""Compatible parameter set construction and refinement for Problem 2.

This module implements the set-based parameter identification framework:
at round k, the compatible set Θ_k contains all parameter vectors that
explain the acquired observations within predeclared tolerances.

Key differences from Problem 1 (ambiguity.py):
- Problem 1: unknown dynamics form, learning F from unpaired snapshots
- Problem 2: known equation architecture, identifying θ in F(x,u;θ)

The compatible set Θ_k is defined as:
    Θ_k = ∩_{c ∈ C_k} {θ ∈ Θ_adm : D_c(M_c(θ), P̂_c) ≤ ε_c}

where C_k is the set of acquired conditions, D_c is a per-condition
discrepancy measure, M_c(θ) is the model prediction, and P̂_c is the
observed distribution (first two replicates for fitting, third for checking).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .baselines import KnownFormRNAODE
from .learning import TrainConfig, fit_unpaired_snapshots
from .metrics import control_adjusted_rmse, sliced_wasserstein
from .protocol import Condition, ReplicatedSnapshot
from .simulation import Family


@dataclass(frozen=True)
class ParameterBounds:
    """Admissible parameter ranges, normalized or with known scale."""

    weights_min: float = -1.2
    weights_max: float = 1.2
    gamma_min: float = 0.5
    gamma_max: float = 1.5
    basal_min: float = 0.1
    basal_max: float = 0.5
    amplitude_min: float = 1.0
    amplitude_max: float = 3.0
    coupling_min: float = 0.0
    coupling_max: float = 1.0
    efficiency_min: float = 0.5
    efficiency_max: float = 1.0

    def __post_init__(self) -> None:
        bounds = [
            (self.weights_min, self.weights_max),
            (self.gamma_min, self.gamma_max),
            (self.basal_min, self.basal_max),
            (self.amplitude_min, self.amplitude_max),
            (self.coupling_min, self.coupling_max),
            (self.efficiency_min, self.efficiency_max),
        ]
        if any(low >= high for low, high in bounds):
            raise ValueError("all parameter bounds must have min < max")


@dataclass(frozen=True)
class ConditionTolerance:
    """Per-condition discrepancy thresholds, frozen before blind networks."""

    condition: Condition
    max_distribution_error: float
    max_effect_error: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_distribution_error) or self.max_distribution_error <= 0:
            raise ValueError("distribution tolerance must be finite and positive")
        if not np.isfinite(self.max_effect_error) or self.max_effect_error <= 0:
            raise ValueError("effect tolerance must be finite and positive")


@dataclass(frozen=True)
class ParameterSample:
    """One point in parameter space with its compatibility status."""

    index: int
    weights: np.ndarray
    gamma: np.ndarray
    basal: np.ndarray
    amplitude: np.ndarray
    coupling: np.ndarray
    efficiencies: np.ndarray
    compatible: bool
    max_violation: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_violation):
            raise ValueError("max_violation must be finite")


class ParameterSetApproximation:
    """Finite sample-based approximation of the compatible parameter set.

    This is not the full Θ_k (which may be uncountable), but a finite grid
    or ensemble that survived all acquired constraints. The approximation
    enables:
    - Volume shrinkage estimation via survival fraction
    - Prediction disagreement via ensemble spread
    - Parameter range estimation via quantiles

    For identifiability, report combinations that appear as products in the
    equations (e.g., amplitude * coupling * weights[i,j]), not raw parameters.
    """

    def __init__(
        self,
        family: Family,
        n_genes: int,
        bounds: ParameterBounds,
        tolerances: Sequence[ConditionTolerance],
        *,
        n_samples: int = 1000,
        seed: int = 0,
    ):
        if family not in ("sigmoid", "hill"):
            raise ValueError("family must be sigmoid or hill")
        if n_genes < 2 or n_samples < 10:
            raise ValueError("need at least 2 genes and 10 samples")

        self.family = family
        self.n_genes = n_genes
        self.bounds = bounds
        self.tolerances = tuple(tolerances)
        self.seed = seed

        self._initial_samples = self._draw_initial_samples(n_samples)
        self._survivors: list[ParameterSample] = list(self._initial_samples)
        self._round = 0

    def _draw_initial_samples(self, n: int) -> list[ParameterSample]:
        """Sobol quasi-random samples from the admissible box."""
        rng = np.random.default_rng(self.seed)

        samples = []
        for index in range(n):
            weights = rng.uniform(
                self.bounds.weights_min,
                self.bounds.weights_max,
                size=(self.n_genes, self.n_genes),
            )
            np.fill_diagonal(weights, 0)

            gamma = rng.uniform(self.bounds.gamma_min, self.bounds.gamma_max, size=self.n_genes)
            basal = rng.uniform(self.bounds.basal_min, self.bounds.basal_max, size=self.n_genes)
            amplitude = rng.uniform(
                self.bounds.amplitude_min, self.bounds.amplitude_max, size=self.n_genes
            )
            coupling = rng.uniform(
                self.bounds.coupling_min, self.bounds.coupling_max, size=self.n_genes
            )
            efficiencies = rng.uniform(
                self.bounds.efficiency_min, self.bounds.efficiency_max, size=self.n_genes
            )

            samples.append(
                ParameterSample(
                    index=index,
                    weights=weights,
                    gamma=gamma,
                    basal=basal,
                    amplitude=amplitude,
                    coupling=coupling,
                    efficiencies=efficiencies,
                    compatible=True,
                    max_violation=0.0,
                )
            )

        return samples

    @property
    def n_initial(self) -> int:
        return len(self._initial_samples)

    @property
    def n_survivors(self) -> int:
        return len(self._survivors)

    @property
    def survival_fraction(self) -> float:
        return self.n_survivors / max(1, self.n_initial)

    @property
    def volume_shrinkage(self) -> float:
        """Approximate volume ratio via survival fraction (assuming uniform prior)."""
        return 1.0 - self.survival_fraction

    def refine(
        self,
        baseline: ReplicatedSnapshot,
        new_observations: Mapping[Condition, ReplicatedSnapshot],
        *,
        step: float = 0.1,
        projection_seed: int = 0,
    ) -> tuple[int, int]:
        """Check each survivor against new constraints, update compatible set.

        Returns (n_eliminated, n_remaining).
        """
        if not new_observations:
            return 0, self.n_survivors

        initial_count = self.n_survivors
        new_tolerances = {tol.condition: tol for tol in self.tolerances}

        new_survivors = []
        for sample in self._survivors:
            max_violation = 0.0
            compatible = True

            for condition, snapshot in new_observations.items():
                if condition not in new_tolerances:
                    continue

                tol = new_tolerances[condition]

                model = self._instantiate_model(sample)
                initial_cells = baseline.fit_treated.reshape(-1, self.n_genes)

                predicted_treated = self._predict(model, initial_cells, condition, step)
                predicted_control = self._predict(
                    model, initial_cells, Condition(None, 0, condition.time), step
                )

                if snapshot.fit_control is None:
                    raise ValueError("compatible set refinement requires matched controls")

                dist_error = sliced_wasserstein(
                    predicted_treated,
                    snapshot.fit_treated.reshape(-1, self.n_genes),
                    seed=projection_seed,
                )

                effect_error = control_adjusted_rmse(
                    predicted_treated,
                    predicted_control,
                    snapshot.fit_treated.reshape(-1, self.n_genes),
                    snapshot.fit_control.reshape(-1, self.n_genes),
                )

                violation = max(
                    dist_error - tol.max_distribution_error,
                    effect_error - tol.max_effect_error,
                )

                max_violation = max(max_violation, violation)

                if violation > 0:
                    compatible = False
                    break

            if compatible:
                new_survivors.append(
                    ParameterSample(
                        index=sample.index,
                        weights=sample.weights,
                        gamma=sample.gamma,
                        basal=sample.basal,
                        amplitude=sample.amplitude,
                        coupling=sample.coupling,
                        efficiencies=sample.efficiencies,
                        compatible=True,
                        max_violation=max_violation,
                    )
                )

        self._survivors = new_survivors
        self._round += 1

        eliminated = initial_count - len(new_survivors)
        return eliminated, len(new_survivors)

    def _instantiate_model(self, sample: ParameterSample) -> KnownFormRNAODE:
        """Create a model with the given parameter values."""
        model = KnownFormRNAODE(self.family, self.n_genes, min_efficiency=self.bounds.efficiency_min)

        with torch.no_grad():
            model.weights.copy_(torch.from_numpy(sample.weights.astype(np.float32)))
            model.gamma_raw.copy_(
                torch.from_numpy(
                    np.log(np.exp(sample.gamma - 1e-8) - 1).astype(np.float32)
                )
            )
            model.basal_raw.copy_(
                torch.from_numpy(
                    np.log(np.exp(sample.basal - 1e-8) - 1).astype(np.float32)
                )
            )
            model.amplitude_raw.copy_(
                torch.from_numpy(
                    np.log(np.exp(sample.amplitude - 1e-8) - 1).astype(np.float32)
                )
            )
            model.coupling_raw.copy_(
                torch.from_numpy(
                    np.log(sample.coupling / (1 - sample.coupling + 1e-8)).astype(np.float32)
                )
            )
            model.rho_raw.copy_(
                torch.from_numpy(
                    np.log(
                        (sample.efficiencies - self.bounds.efficiency_min)
                        / (1 - sample.efficiencies + 1e-8)
                    ).astype(np.float32)
                )
            )

        return model

    def _predict(
        self,
        model: KnownFormRNAODE,
        baseline_cells: np.ndarray,
        condition: Condition,
        step: float,
    ) -> np.ndarray:
        """Predict with one parameter sample."""
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        initial = torch.as_tensor(baseline_cells, device=device, dtype=dtype)

        rho_override = None
        if condition.target is not None:
            idx = condition.target
            rho_override = float(model.efficiencies[idx].detach().cpu().numpy())

        result = model.integrate(
            initial, condition.time, condition.target, condition.dose, step=step, rho_override=rho_override
        )

        return result.detach().cpu().numpy()

    def prediction_disagreement(
        self,
        baseline: ReplicatedSnapshot,
        condition: Condition,
        *,
        step: float = 0.1,
        quantiles: tuple[float, float] = (0.025, 0.975),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction bounds from surviving ensemble.

        Each survivor predicts a population; quantiles are computed gene-wise
        across the ensemble means. This is prediction disagreement, not a
        calibrated confidence interval.
        """
        if not self._survivors:
            raise RuntimeError("no compatible parameters remain")

        initial_cells = baseline.fit_treated.reshape(-1, self.n_genes)

        predictions = []
        for sample in self._survivors:
            model = self._instantiate_model(sample)
            pred = self._predict(model, initial_cells, condition, step)
            predictions.append(pred.mean(axis=0))

        ensemble = np.stack(predictions, axis=0)
        lower = np.quantile(ensemble, quantiles[0], axis=0)
        upper = np.quantile(ensemble, quantiles[1], axis=0)

        return lower, upper

    def parameter_ranges(self) -> dict[str, tuple[float, float]]:
        """Return (min, max) for identifiable combinations across survivors."""
        if not self._survivors:
            return {}

        ranges = {}

        all_weights = np.stack([s.weights for s in self._survivors], axis=0)
        ranges["weights_l1_norm"] = (
            float(np.abs(all_weights).sum(axis=(1, 2)).min()),
            float(np.abs(all_weights).sum(axis=(1, 2)).max()),
        )

        all_gamma = np.stack([s.gamma for s in self._survivors], axis=0)
        ranges["gamma_mean"] = (float(all_gamma.mean(axis=1).min()), float(all_gamma.mean(axis=1).max()))

        all_amplitude = np.stack([s.amplitude for s in self._survivors], axis=0)
        all_coupling = np.stack([s.coupling for s in self._survivors], axis=0)
        effective_strength = all_amplitude * all_coupling
        ranges["amplitude_x_coupling"] = (
            float(effective_strength.mean(axis=1).min()),
            float(effective_strength.mean(axis=1).max()),
        )

        all_eff = np.stack([s.efficiencies for s in self._survivors], axis=0)
        ranges["efficiency_median"] = (
            float(np.median(all_eff, axis=1).min()),
            float(np.median(all_eff, axis=1).max()),
        )

        return ranges


def choose_parameter_informed(
    candidate_pool: tuple[Condition, ...],
    acquired: tuple[Condition, ...],
    parameter_set: ParameterSetApproximation,
    baseline: ReplicatedSnapshot,
    *,
    step: float = 0.1,
) -> Condition:
    """Select the candidate condition maximizing prediction disagreement.

    This is the Problem 2 active selection strategy: choose experiments that
    reveal which parameters are compatible, not which edge structures.
    """
    if not candidate_pool:
        raise ValueError("candidate pool is empty")

    available = [c for c in candidate_pool if c not in acquired]
    if not available:
        raise ValueError("all candidates already acquired")

    disagreements = []
    for condition in available:
        lower, upper = parameter_set.prediction_disagreement(baseline, condition, step=step)
        disagreement = float(np.mean(upper - lower))
        disagreements.append((disagreement, condition))

    disagreements.sort(reverse=True)
    return disagreements[0][1]
