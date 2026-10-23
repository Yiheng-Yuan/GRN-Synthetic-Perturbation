"""Observation-only comparators and descriptive scores for the Q1 pilot.

These mean-shift comparators do not model trajectories, recover a GRN, or
infer paired cells. Only two fitting replicates may supply their effects.
Scores concern one held-out replicate; they are not independent-network
significance tests or evidence of identifiable dynamics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np

from .protocol import Condition, ReplicatedSnapshot


FIT_TIMES = (0.25, 4.0)


def _population(values: np.ndarray, name: str) -> np.ndarray:
    population = np.asarray(values, dtype=np.float64)
    if population.ndim != 2 or min(population.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape (cells, genes)")
    if not np.isfinite(population).all() or (population < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative RNA counts")
    return population.copy()


def _sample_baseline(baseline_fit: np.ndarray, cells: int, seed: int) -> np.ndarray:
    if isinstance(cells, bool) or not isinstance(cells, Integral) or cells < 1:
        raise ValueError("cells must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    baseline = _population(baseline_fit, "baseline_fit")
    indices = np.random.default_rng(int(seed)).integers(0, len(baseline), size=int(cells))
    return baseline[indices].copy()


class NoResponseBaseline:
    """An empirical fit-baseline population with zero perturbation response.

    ``baseline_fit`` must be flattened from the first two baseline replicates
    by the caller, never from the check replicate. ``observations`` is accepted
    for API symmetry but is not read by this no-response comparator.
    """

    @classmethod
    def fit(
        cls,
        baseline_fit: np.ndarray,
        observations: Mapping[Condition, ReplicatedSnapshot] | None = None,
        *,
        cells: int = 128,
        seed: int = 0,
    ) -> NoResponseBaseline:
        instance = cls()
        instance._control = _sample_baseline(baseline_fit, cells, seed)
        instance.n_genes = instance._control.shape[1]
        return instance

    def predict(self, condition: Condition) -> tuple[np.ndarray, np.ndarray]:
        self._validate_condition(condition)
        return self._control.copy(), self._control.copy()

    def prediction_note(self, condition: Condition) -> str:
        self._validate_condition(condition)
        return "no_response"

    def _validate_condition(self, condition: Condition) -> None:
        if not hasattr(self, "_control"):
            raise RuntimeError("fit the baseline before prediction")
        if not isinstance(condition, Condition):
            raise TypeError("condition must be a Condition")
        if condition.target is not None and condition.target >= self.n_genes:
            raise ValueError("condition target is outside the observed gene dimension")


class LinearEffectBaseline(NoResponseBaseline):
    """Piecewise-linear raw-count mean effects with an empirical population.

    For each observed (target, dose), use only t=0.25 and t=4 fit replicates,
    with a zero-effect anchor at t=0. No dose or unseen-target transfer is
    claimed. Beyond t=4 the last effect is held, explicitly not mechanistic
    extrapolation. Nonnegative clipping can alter the realized mean shift;
    scores always use the actual clipped predictions.

    The same empirical baseline sample supplies both arm distributions. This
    numerical construction does not assert pairing between observed cells.
    """

    @classmethod
    def fit(
        cls,
        baseline_fit: np.ndarray,
        observations: Mapping[Condition, ReplicatedSnapshot],
        *,
        cells: int = 128,
        seed: int = 0,
    ) -> LinearEffectBaseline:
        instance = super().fit(baseline_fit, cells=cells, seed=seed)
        if not observations:
            raise ValueError("at least one target/dose with both fit times is required")
        grouped: dict[tuple[int, float], dict[float, np.ndarray]] = {}
        for condition, snapshot in observations.items():
            instance._validate_condition(condition)
            if condition.target is None or condition.time not in FIT_TIMES:
                raise ValueError("linear comparator accepts only targeted t=0.25 and t=4 fit conditions")
            if snapshot.treated.shape[-1] != instance.n_genes:
                raise ValueError("snapshot gene dimension differs from baseline")
            fit_control = snapshot.fit_control
            if fit_control is None:
                raise ValueError("matched controls are required for targeted observations")
            # Independent arm means: do not pair rows or inspect replicate 2.
            effect = snapshot.fit_treated.mean(axis=(0, 1)) - fit_control.mean(axis=(0, 1))
            grouped.setdefault((condition.target, condition.dose), {})[condition.time] = effect
        instance._effects = {}
        for key, by_time in grouped.items():
            if set(by_time) != set(FIT_TIMES):
                raise ValueError("every fitted target/dose must have both t=0.25 and t=4")
            instance._effects[key] = np.stack(
                [np.zeros(instance.n_genes), by_time[FIT_TIMES[0]], by_time[FIT_TIMES[1]]]
            )
        return instance

    def predict(self, condition: Condition) -> tuple[np.ndarray, np.ndarray]:
        self._validate_condition(condition)
        effect = np.zeros(self.n_genes)
        key = (condition.target, condition.dose)
        if key in self._effects:
            effect = np.array([
                np.interp(condition.time, (0.0, *FIT_TIMES), self._effects[key][:, gene])
                for gene in range(self.n_genes)
            ])
        return np.maximum(self._control + effect, 0.0), self._control.copy()

    def prediction_note(self, condition: Condition) -> str:
        self._validate_condition(condition)
        if condition.target is None:
            return "untreated_baseline"
        if not any(target == condition.target for target, _ in self._effects):
            return "unseen_target_no_response"
        if (condition.target, condition.dose) not in self._effects:
            return "unseen_dose_no_response"
        if condition.time > FIT_TIMES[-1]:
            return "late_time_endpoint_hold"
        return "piecewise_linear_mean_effect"


class DirectOnlyTransferBaseline(NoResponseBaseline):
    """Transfer the average standardized self-response, never downstream effects.

    Fit only on supplied training targets and the first two replicates. For
    each dose/time, average their target effects in baseline-standard-deviation
    units. Apply that curve to a requested gene using its baseline scale. This
    is a deliberately weak observation-only comparator, not a GRN estimate.
    """

    @classmethod
    def fit(
        cls,
        baseline_fit: np.ndarray,
        observations: Mapping[Condition, ReplicatedSnapshot],
        *,
        cells: int = 128,
        seed: int = 0,
        times: Sequence[float] = (0.25, 1.0, 2.0, 4.0, 8.0, 48.0),
    ) -> DirectOnlyTransferBaseline:
        instance = super().fit(baseline_fit, cells=cells, seed=seed)
        if not observations:
            raise ValueError("at least one training target is required")
        try:
            grid = tuple(times)
        except TypeError:
            raise ValueError("times must be a sequence of sampling times") from None
        if (not grid or any(isinstance(t, bool) or not isinstance(t, Real)
                            or not np.isfinite(t) or t <= 0 for t in grid)):
            raise ValueError("times must be finite positive sampling times")
        instance._times = tuple(float(t) for t in grid)
        if any(b <= a for a, b in zip(instance._times, instance._times[1:])):
            raise ValueError("times must be strictly increasing")
        instance._scale = np.maximum(_population(baseline_fit, "baseline_fit").std(axis=0), 1.0)
        effects: dict[tuple[int, float], dict[float, float]] = {}
        for condition, snapshot in observations.items():
            instance._validate_condition(condition)
            if condition.target is None or condition.time not in instance._times:
                raise ValueError("only targeted conditions on the registered time grid are allowed")
            if snapshot.treated.shape[-1] != instance.n_genes:
                raise ValueError("snapshot gene dimension differs from baseline")
            control = snapshot.fit_control
            if control is None:
                raise ValueError("matched controls are required for targeted observations")
            target = condition.target
            delta = snapshot.fit_treated[..., target].mean() - control[..., target].mean()
            effects.setdefault((target, condition.dose), {})[condition.time] = float(delta / instance._scale[target])
        targets = sorted({target for target, _ in effects})
        doses = sorted({dose for _, dose in effects})
        if any(set(effects.get((target, dose), {})) != set(instance._times)
               for target in targets for dose in doses):
            raise ValueError("every training target/dose must have the complete common time grid")
        instance._curves = {
            dose: np.array([0.0, *(np.mean([effects[(q, dose)][t] for q in targets])
                                    for t in instance._times)]) for dose in doses
        }
        return instance

    def predict(self, condition: Condition) -> tuple[np.ndarray, np.ndarray]:
        self._validate_condition(condition)
        treated = self._control.copy()
        if condition.target is not None and condition.dose in self._curves:
            effect = np.interp(condition.time, (0.0, *self._times), self._curves[condition.dose])
            target = condition.target
            # Clipping may change the realized mean shift; score these actual
            # nonnegative populations rather than the unclipped template.
            treated[:, target] = np.maximum(treated[:, target] + effect * self._scale[target], 0.0)
        return treated, self._control.copy()

    def prediction_note(self, condition: Condition) -> str:
        self._validate_condition(condition)
        if condition.target is None:
            return "untreated_baseline"
        if condition.dose not in self._curves:
            return "unseen_dose_no_response"
        if condition.time > self._times[-1]:
            return "direct_only_transfer_endpoint_hold"
        return "direct_only_standardized_transfer"


def marginal_wasserstein1(left: np.ndarray, right: np.ndarray) -> float:
    """Exact average 1D empirical Wasserstein-1 in raw count units.

    Each cell has equal weight within its own population; differing sample
    counts are supported. Marginal distances do not assess gene covariance.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (left.ndim != 2 or right.ndim != 2 or min(left.shape) < 1 or
            min(right.shape) < 1 or left.shape[1] != right.shape[1]):
        raise ValueError("distance inputs require nonempty cells-by-genes matrices with matching genes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("distance inputs must be finite")
    # Integrate the piecewise-constant empirical quantile functions exactly.
    breaks = np.unique(np.concatenate((
        np.arange(len(left) + 1) / len(left),
        np.arange(len(right) + 1) / len(right),
    )))
    midpoints = (breaks[1:] + breaks[:-1]) / 2
    left_q = np.quantile(left, midpoints, axis=0, method="inverted_cdf")
    right_q = np.quantile(right, midpoints, axis=0, method="inverted_cdf")
    return float(np.mean(np.sum(np.diff(breaks)[:, None] * np.abs(left_q - right_q), axis=0)))


def score_prediction(
    treated_prediction: np.ndarray,
    control_prediction: np.ndarray,
    condition: Condition,
    snapshot: ReplicatedSnapshot,
    *,
    replicate: int = 2,
    directions: np.ndarray | None = None,
) -> dict:
    """Score unpaired populations without hidden topology or kinetic truth.

    ``non_target`` means all genes except the perturbed gene, NOT established
    downstream genes. Mean errors and distribution errors remain separate.
    Optional fixed projection directions have shape (directions, genes).
    """
    treated = _population(treated_prediction, "treated_prediction")
    control = _population(control_prediction, "control_prediction")
    n_genes = snapshot.treated.shape[-1]
    if treated.shape[1] != n_genes or control.shape[1] != n_genes:
        raise ValueError("prediction and observation gene dimensions must match")
    if isinstance(replicate, bool) or not isinstance(replicate, Integral) or not 0 <= replicate < 3:
        raise ValueError("replicate must be 0, 1 or 2 (the default check replicate)")
    if condition.target is not None and condition.target >= n_genes:
        raise ValueError("condition target is outside the observed gene dimension")
    observed_treated = snapshot.treated[int(replicate)]
    if snapshot.matched_control is None:
        if condition.target is not None:
            raise ValueError("a targeted observation must have matched control")
        observed_control = observed_treated
    else:
        observed_control = snapshot.matched_control[int(replicate)]
    predicted_delta = treated.mean(axis=0) - control.mean(axis=0)
    observed_delta = observed_treated.mean(axis=0) - observed_control.mean(axis=0)
    delta_error = predicted_delta - observed_delta
    non_target = np.ones(n_genes, dtype=bool)
    if condition.target is not None:
        non_target[condition.target] = False
    target = condition.target
    result = {
        "target": target,
        "dose": float(condition.dose),
        "time": float(condition.time),
        "replicate": int(replicate),
        "n_genes": int(n_genes),
        "mean_rmse": float(np.sqrt(np.mean((treated.mean(axis=0) - observed_treated.mean(axis=0)) ** 2))),
        "delta_rmse_all": float(np.sqrt(np.mean(delta_error ** 2))),
        # One target per condition means absolute error; averaging across
        # conditions is MAE. Retain the historical name as a numeric alias.
        "delta_rmse_target": None if target is None else float(abs(delta_error[target])),
        "target_effect_mae": None if target is None else float(abs(delta_error[target])),
        "delta_rmse_non_target": None if not non_target.any() else float(np.sqrt(np.mean(delta_error[non_target] ** 2))),
        "observed_abs_target_effect": None if target is None else float(abs(observed_delta[target])),
        "observed_target_effect": None if target is None else float(observed_delta[target]),
        "pred_target_effect": None if target is None else float(predicted_delta[target]),
        "treated_marginal_wasserstein1": marginal_wasserstein1(treated, observed_treated),
        "control_marginal_wasserstein1": marginal_wasserstein1(control, observed_control),
    }
    if directions is not None:
        fixed = np.asarray(directions, dtype=np.float64)
        if (fixed.ndim != 2 or fixed.shape[1] != n_genes or len(fixed) < 1 or
                not np.isfinite(fixed).all()):
            raise ValueError("directions must be a finite nonempty directions-by-genes matrix")
        norms = np.linalg.norm(fixed, axis=1)
        if (norms == 0).any():
            raise ValueError("projection directions cannot have zero norm")
        fixed = fixed / norms[:, None]
        result["treated_sliced_wasserstein1"] = marginal_wasserstein1(treated @ fixed.T, observed_treated @ fixed.T)
    return result


def aggregate_scores(scores: Sequence[Mapping]) -> dict:
    """Condition-average descriptive errors, not independent-network inference."""
    if not scores:
        raise ValueError("at least one scored condition is required")
    metric_names = (
        "mean_rmse", "delta_rmse_all", "delta_rmse_target", "target_effect_mae", "delta_rmse_non_target",
        "treated_marginal_wasserstein1", "control_marginal_wasserstein1",
        "treated_sliced_wasserstein1",
    )
    means, counts = {}, {}
    for name in metric_names:
        values = [float(score[name]) for score in scores if score.get(name) is not None]
        if values:
            if not np.isfinite(values).all():
                raise ValueError("score values must be finite")
            means[name] = float(np.mean(values))
            counts[name] = len(values)
    return {"n_conditions": len(scores), "mean_metrics": means, "metric_condition_counts": counts}
