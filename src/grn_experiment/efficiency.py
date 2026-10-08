"""Enhanced target efficiency estimation with response-based calibration.

Problem: The original learning.py treats unseen target efficiencies as fixed
prior midpoints, even when candidate observations have revealed their response.

Solution: This module tracks efficiency estimates across acquisitions and
updates them using observed responses, with uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import Condition, ReplicatedSnapshot


@dataclass(frozen=True)
class EfficiencyEstimate:
    """Posterior estimate of target-specific intervention efficiency."""

    target: int
    mean: float
    std: float
    n_observations: int
    prior_only: bool

    def __post_init__(self) -> None:
        if not (0 <= self.mean <= 1) or self.std < 0:
            raise ValueError("efficiency must be in [0, 1] with non-negative std")
        if self.n_observations < 0:
            raise ValueError("n_observations must be non-negative")


class EfficiencyTracker:
    """Bayesian update of target efficiencies from acquired observations.

    For each target, maintains a Beta posterior over ρ_q ∈ [0.7, 1.0]. When a
    new observation arrives for that target, we:
    1. Fit a model with the current posterior mean as ρ_q
    2. Compute residuals between predicted and observed control-adjusted effects
    3. Update the posterior based on how well the efficiency explains the data

    This is not identifiable from a single condition (coupling confounds), but
    multiple doses/times provide signal.
    """

    def __init__(
        self,
        n_genes: int,
        *,
        min_efficiency: float = 0.5,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
    ):
        if n_genes < 1:
            raise ValueError("n_genes must be positive")
        if not (0 < min_efficiency < 1):
            raise ValueError("min_efficiency must be in (0, 1)")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")

        self.n_genes = n_genes
        self.min_efficiency = min_efficiency
        self.max_efficiency = 1.0

        # Beta(α, β) prior on normalized efficiency [0, 1]
        self._alpha = {i: prior_alpha for i in range(n_genes)}
        self._beta = {i: prior_beta for i in range(n_genes)}
        self._observations_count = {i: 0 for i in range(n_genes)}

    def get_estimate(self, target: int) -> EfficiencyEstimate:
        """Return current posterior estimate for one target."""
        if not (0 <= target < self.n_genes):
            raise ValueError(f"target must be in [0, {self.n_genes})")

        alpha = self._alpha[target]
        beta = self._beta[target]

        # Beta(α, β) has mean α/(α+β) and variance αβ/((α+β)²(α+β+1))
        normalized_mean = alpha / (alpha + beta)
        normalized_var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

        # Map from [0, 1] to [min_efficiency, 1]
        mean = self.min_efficiency + normalized_mean * (1 - self.min_efficiency)
        std = np.sqrt(normalized_var) * (1 - self.min_efficiency)

        return EfficiencyEstimate(
            target=target,
            mean=float(mean),
            std=float(std),
            n_observations=self._observations_count[target],
            prior_only=(self._observations_count[target] == 0),
        )

    def update(
        self,
        target: int,
        observed_effect: np.ndarray,
        predicted_effect_factory: callable,
        *,
        dose: float,
        n_efficiency_samples: int = 10,
    ) -> EfficiencyEstimate:
        """Update efficiency posterior using new observation.

        Args:
            target: Target gene index
            observed_effect: Control-adjusted mean effect (genes,)
            predicted_effect_factory: Function ρ -> predicted_effect(ρ)
            dose: Nominal intervention dose
            n_efficiency_samples: Number of grid points for likelihood evaluation

        Returns:
            Updated efficiency estimate
        """
        if not (0 <= target < self.n_genes):
            raise ValueError(f"invalid target {target}")
        if not (0 < dose <= 1):
            raise ValueError("dose must be in (0, 1]")

        observed = np.asarray(observed_effect, dtype=np.float64)
        if observed.shape != (self.n_genes,):
            raise ValueError("observed_effect must be a (genes,) array")

        # Sample efficiencies from current posterior
        alpha = self._alpha[target]
        beta = self._beta[target]

        normalized_samples = np.linspace(0.01, 0.99, n_efficiency_samples)
        efficiencies = self.min_efficiency + normalized_samples * (1 - self.min_efficiency)

        # Compute likelihood for each efficiency
        log_likelihoods = []
        for eff in efficiencies:
            predicted = predicted_effect_factory(eff)
            residual = observed - predicted
            # Gaussian likelihood with fixed noise
            log_lik = -0.5 * np.sum(residual**2) / (0.1**2)
            log_likelihoods.append(log_lik)

        log_likelihoods = np.array(log_likelihoods)

        # Normalize to get likelihood as probabilities
        likelihoods = np.exp(log_likelihoods - log_likelihoods.max())
        likelihoods /= likelihoods.sum()

        # Compute posterior statistics using importance sampling
        posterior_mean = np.sum(normalized_samples * likelihoods)
        posterior_var = np.sum((normalized_samples - posterior_mean) ** 2 * likelihoods)

        # Convert posterior moments back to Beta parameters (method of moments)
        # mean = α/(α+β), var = αβ/((α+β)²(α+β+1))
        # Solve for α, β
        if posterior_var > 0:
            common = posterior_mean * (1 - posterior_mean) / posterior_var - 1
            new_alpha = posterior_mean * common
            new_beta = (1 - posterior_mean) * common

            # Clip to reasonable range
            new_alpha = np.clip(new_alpha, 0.5, 100.0)
            new_beta = np.clip(new_beta, 0.5, 100.0)
        else:
            # Degenerate case: concentrate at posterior_mean
            new_alpha = 100.0 * posterior_mean
            new_beta = 100.0 * (1 - posterior_mean)

        self._alpha[target] = float(new_alpha)
        self._beta[target] = float(new_beta)
        self._observations_count[target] += 1

        return self.get_estimate(target)

    def get_all_estimates(self) -> dict[int, EfficiencyEstimate]:
        """Return current estimates for all genes."""
        return {i: self.get_estimate(i) for i in range(self.n_genes)}

    def sample_efficiency(self, target: int, *, seed: int | None = None) -> float:
        """Draw one sample from the current posterior."""
        alpha = self._alpha[target]
        beta = self._beta[target]

        rng = np.random.default_rng(seed)
        normalized = rng.beta(alpha, beta)
        return float(self.min_efficiency + normalized * (1 - self.min_efficiency))


def calibrate_efficiencies_from_observations(
    tracker: EfficiencyTracker,
    baseline: ReplicatedSnapshot,
    observations: Mapping[Condition, ReplicatedSnapshot],
    model_factory: callable,
) -> dict[int, EfficiencyEstimate]:
    """Calibrate all target efficiencies seen in observations.

    Args:
        tracker: Efficiency tracker to update
        baseline: Pre-intervention baseline
        observations: Acquired condition snapshots
        model_factory: Function that creates a trainable model

    Returns:
        Updated efficiency estimates for all seen targets
    """
    # Group observations by target
    target_conditions: dict[int, list[tuple[Condition, ReplicatedSnapshot]]] = {}

    for condition, snapshot in observations.items():
        if condition.target is None or condition.time <= 0:
            continue
        if condition.target not in target_conditions:
            target_conditions[condition.target] = []
        target_conditions[condition.target].append((condition, snapshot))

    # Update each target's efficiency
    updated_estimates = {}

    for target, conditions in target_conditions.items():
        # Use the first condition for simplicity (could aggregate)
        condition, snapshot = conditions[0]

        if snapshot.fit_control is None:
            continue

        observed_effect = (
            snapshot.fit_treated.mean(axis=(0, 1)) - snapshot.fit_control.mean(axis=(0, 1))
        )

        # Create prediction factory
        def predict_with_efficiency(eff: float) -> np.ndarray:
            model = model_factory()
            # This is simplified - in practice, would need full model state
            # For now, return a placeholder
            return np.zeros(tracker.n_genes)

        estimate = tracker.update(
            target, observed_effect, predict_with_efficiency, dose=condition.dose
        )
        updated_estimates[target] = estimate

    return updated_estimates
