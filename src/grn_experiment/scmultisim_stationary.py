"""Finite-budget stationary and numerical diagnostics for the scMultiSim extension.

The reports produced here are numerical diagnostics, not proofs of ergodicity,
uniqueness, or calibrated statistical tests.  Thresholds are explicit development
choices and must be frozen before a blind experiment.  In particular, a requested
sampling time is never automatically classified as a stationary endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from .scmultisim import CellState, KineticParameters, Simulator, parameter_digest


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _positive_int(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class StationarityConfig:
    """Development diagnostics; ``diagnostic_cells`` is per independent start.

    Each window pools ``samples_per_window`` snapshots of each ensemble.  The
    samples of a lineage within a window are correlated, so the reported window
    count must not be interpreted as the number of independent cells.
    """

    min_time: float = 16.0
    max_time: float = 128.0
    window: float = 8.0
    consecutive_windows: int = 3
    mean_rtol: float = 0.15
    variance_rtol: float = 0.25
    distribution_tol: float = 0.2
    atol: float = 0.1
    diagnostic_cells: int = 256
    samples_per_window: int = 3

    def __post_init__(self) -> None:
        for name in ("min_time", "mean_rtol", "variance_rtol", "distribution_tol"):
            _positive_finite(getattr(self, name), name, allow_zero=True)
        for name in ("max_time", "window", "atol"):
            _positive_finite(getattr(self, name), name)
        for name in ("consecutive_windows", "samples_per_window"):
            _positive_int(getattr(self, name), name)
        _positive_int(self.diagnostic_cells, "diagnostic_cells", minimum=2)
        if self.max_time < self.min_time + 2 * self.window:
            raise ValueError("max_time must allow at least two complete windows after min_time")


@dataclass(frozen=True)
class StationarityResult:
    passed: bool
    burnin: float | None
    elapsed_time: float
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "burnin": self.burnin,
            "elapsed_time": self.elapsed_time,
            "report": self.report,
        }


def _state_array(state: CellState) -> np.ndarray:
    return np.concatenate((state.G, state.U, state.M), axis=1).astype(float, copy=False)


def _distribution_array(value: np.ndarray | CellState) -> np.ndarray:
    array = _state_array(value) if isinstance(value, CellState) else np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("distribution samples must be a nonempty 2D array with at least two rows")
    if not np.isfinite(array).all():
        raise ValueError("distribution samples must be finite")
    return array


def _wasserstein_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact empirical 1D Wasserstein distances, separately by column."""
    if left.shape[0] == right.shape[0]:
        return np.mean(np.abs(np.sort(left, axis=0) - np.sort(right, axis=0)), axis=0)
    distances = []
    for column in range(left.shape[1]):
        x = np.sort(left[:, column])
        y = np.sort(right[:, column])
        support = np.sort(np.concatenate((x, y)))
        widths = np.diff(support)
        cdf_x = np.searchsorted(x, support[:-1], side="right") / len(x)
        cdf_y = np.searchsorted(y, support[:-1], side="right") / len(y)
        distances.append(float(np.sum(widths * np.abs(cdf_x - cdf_y))))
    return np.asarray(distances)


def compare_distributions(
    left: np.ndarray | CellState,
    right: np.ndarray | CellState,
    *,
    mean_rtol: float = 0.15,
    variance_rtol: float = 0.25,
    distribution_tol: float = 0.2,
    atol: float = 0.1,
) -> dict[str, Any]:
    """Compare moments, marginals, and fixed projections of the joint state.

    Means and variances use symmetric absolute-plus-relative tolerances.  The
    distribution criterion is the larger of maximum marginal and maximum
    projected empirical Wasserstein-1 distances, standardized per feature by
    the pooled standard deviation, mean magnitude, or ``atol``.  Fixed seeded
    projections also test joint dependence, but cannot prove equality of the
    joint distributions.  This function supplies no p-value or confidence level.
    """
    for name, value in (("mean_rtol", mean_rtol), ("variance_rtol", variance_rtol),
                        ("distribution_tol", distribution_tol)):
        _positive_finite(value, name, allow_zero=True)
    _positive_finite(atol, "atol")
    a = _distribution_array(left)
    b = _distribution_array(right)
    if a.shape[1] != b.shape[1]:
        raise ValueError("sample features must match")
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    mean_error = np.abs(ma - mb)
    variance_error = np.abs(va - vb)
    mean_limits = atol + mean_rtol * np.maximum(np.abs(ma), np.abs(mb))
    variance_limits = atol**2 + variance_rtol * np.maximum(va, vb)
    scale = np.maximum.reduce((np.full_like(ma, atol), np.sqrt((va + vb) / 2),
                               (np.abs(ma) + np.abs(mb)) / 2))
    marginal = _wasserstein_columns(a, b) / scale
    projection_rng = np.random.default_rng(537021)
    directions = projection_rng.normal(size=(a.shape[1], 32))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    projected = _wasserstein_columns((a / scale) @ directions, (b / scale) @ directions)
    max_marginal, max_projected = float(marginal.max()), float(projected.max())
    passed = bool(np.all(mean_error <= mean_limits) and np.all(variance_error <= variance_limits)
                  and max(max_marginal, max_projected) <= distribution_tol)
    return {
        "passed": passed,
        "mean_tolerance_ratio_max": float(np.max(mean_error / mean_limits)),
        "variance_tolerance_ratio_max": float(np.max(variance_error / variance_limits)),
        "max_standardized_marginal_wasserstein": max_marginal,
        "max_standardized_projected_wasserstein": max_projected,
        "left_rows": int(len(a)),
        "right_rows": int(len(b)),
        "features": int(a.shape[1]),
        "joint_projection_seed": 537021,
        "joint_projection_count": 32,
        "confidence_statement": "finite_sample_diagnostic_not_a_calibrated_hypothesis_test",
    }


def _comparison_kwargs(config: StationarityConfig) -> dict[str, float]:
    return {name: getattr(config, name)
            for name in ("mean_rtol", "variance_rtol", "distribution_tol", "atol")}


def _fresh_start(parameters: KineticParameters, n_cells: int, seed: int) -> CellState:
    rng = np.random.default_rng(seed)
    p_on = np.asarray(parameters.kon) / (np.asarray(parameters.kon) + np.asarray(parameters.koff))
    return CellState(
        G=(rng.random((n_cells, parameters.n_genes)) < p_on).astype(np.int8),
        U=np.zeros((n_cells, parameters.n_genes)),
        M=np.zeros((n_cells, parameters.n_genes)),
    )


def sample_initial_states(
    parameters: KineticParameters,
    n_cells: int,
    seed: int,
    tau: float,
    burnin: float,
) -> CellState:
    """Independently evolve fresh unperturbed lineages for an accepted burn-in.

    The caller must obtain ``burnin`` from an accepted diagnostic/frozen rule.
    This function does not certify stationarity.  It never resamples a stored
    pool, and the diagnostic ensembles are not recycled into production cells.
    """
    _positive_int(n_cells, "n_cells")
    _positive_finite(tau, "tau")
    _positive_finite(burnin, "burnin")
    streams = np.random.SeedSequence(seed).spawn(2)
    initial = _fresh_start(parameters, n_cells, int(streams[0].generate_state(1, dtype=np.uint64)[0]))
    simulator = Simulator(parameters, initial, seed=int(streams[1].generate_state(1, dtype=np.uint64)[0]), tau=tau)
    return simulator.advance(burnin)


def assess_stationarity(
    parameters: KineticParameters,
    seed: int,
    tau: float,
    config: StationarityConfig | None = None,
    target: int | None = None,
    dose: float = 0.0,
) -> StationarityResult:
    """Assess two independently simulated ensembles from contrasting G/U/M starts.

    All-off/zero and all-on/high-RNA starts are compared within contiguous time
    windows and with each other.  The reported burn-in is accepted only after
    the requested number of consecutive successful window comparisons.  Failure
    returns ``burnin=None`` and retains the diagnostics; no failed network is
    replaced and no endpoint is relabelled as a stationary sample.
    """
    config = StationarityConfig() if config is None else config
    _positive_finite(tau, "tau")
    n, genes = config.diagnostic_cells, parameters.n_genes
    zero = np.zeros((n, genes))
    reference_s = np.asarray(parameters.production(np.zeros((1, genes))), dtype=float).reshape(-1)
    high_u = 10 * np.maximum(1.0, reference_s / np.asarray(parameters.beta))
    high_m = 10 * np.maximum(1.0, reference_s / np.asarray(parameters.gamma))
    starts = [CellState(G=np.zeros_like(zero, dtype=np.int8), U=zero.copy(), M=zero.copy()),
              CellState(G=np.ones_like(zero, dtype=np.int8),
                        U=np.broadcast_to(high_u, (n, genes)).copy(),
                        M=np.broadcast_to(high_m, (n, genes)).copy())]
    streams = np.random.SeedSequence(seed).spawn(2)
    simulators = [Simulator(parameters, state,
                            seed=int(stream.generate_state(1, dtype=np.uint64)[0]),
                            tau=tau, target=target, dose=dose)
                  for state, stream in zip(starts, streams, strict=True)]
    for simulator in simulators:
        simulator.advance(config.min_time)
    previous: list[np.ndarray] | None = None
    records: list[dict[str, Any]] = []
    streak = 0
    elapsed = float(config.min_time)
    accepted = False
    window_count = int(math.floor((config.max_time - config.min_time) / config.window + 1e-12))
    for window_index in range(window_count):
        begin = config.min_time + window_index * config.window
        times = begin + config.window * np.arange(1, config.samples_per_window + 1) / config.samples_per_window
        current = [np.concatenate([_state_array(simulator.advance(float(t))) for t in times], axis=0)
                   for simulator in simulators]
        elapsed = float(times[-1])
        between = compare_distributions(current[0], current[1], **_comparison_kwargs(config))
        within = ([] if previous is None else
                  [compare_distributions(old, new, **_comparison_kwargs(config))
                   for old, new in zip(previous, current, strict=True)])
        passed = bool(previous is not None and between["passed"] and all(item["passed"] for item in within))
        streak = streak + 1 if passed else 0
        records.append({"window_start": float(begin), "window_end": elapsed,
                        "between_starts": between, "within_starts": within,
                        "passed": passed, "consecutive_successes": streak})
        previous = current
        if streak >= config.consecutive_windows:
            accepted = True
            break
    report = {
        "schema_version": "scmultisim-stationarity-v1",
        "parameter_sha256": parameter_digest(parameters),
        "status": "diagnostic_passed" if accepted else "stationarity_not_established",
        "reason": "consecutive_windows_passed" if accepted else "budget_exhausted",
        "interpretation": "finite_budget_evidence_not_proof_of_unique_stationary_distribution",
        "config": asdict(config), "tau": float(tau), "seed": int(seed),
        "target": None if target is None else int(target), "dose": float(dose),
        "starts": ["all_off_zero_RNA", "all_on_high_RNA"],
        "state_features": "joint_G_U_M",
        "independent_lineages_per_start": n,
        "within_lineage_window_samples_are_correlated": True,
        "diagnostic_samples_used_for_generation": False,
        "windows": records,
    }
    return StationarityResult(accepted, elapsed if accepted else None, elapsed, report)


def analytic_no_feedback_moments(
    parameters: KineticParameters, *, target: int | None = None, dose: float = 0.0,
) -> dict[str, np.ndarray]:
    """Exact stationary first/second moments for the no-edge fluid-RNA process.

    No extra molecule-count shot noise is added: the extension evolves continuous
    U/M between stochastic promoter switches.  These moments refer to U/M before
    the one-time observation floor operation.
    """
    if np.any(np.asarray(parameters.A) != 0):
        raise ValueError("analytic moments require a no-feedback network (A == 0)")
    s = np.asarray(parameters.production(np.zeros((1, parameters.n_genes)), target=target, dose=dose)).reshape(-1)
    beta, gamma = np.asarray(parameters.beta), np.asarray(parameters.gamma)
    rate = np.asarray(parameters.kon) + np.asarray(parameters.koff)
    p = np.asarray(parameters.kon) / rate
    var_g = p * (1 - p)
    cov_gu = s * var_g / (rate + beta)
    cov_gm = beta * cov_gu / (rate + gamma)
    var_u = s * cov_gu / beta
    cov_um = (s * cov_gm + beta * var_u) / (beta + gamma)
    return {"mean_G": p, "mean_U": s * p / beta, "mean_M": s * p / gamma,
            "variance_G": var_g, "variance_U": var_u, "variance_M": beta * cov_um / gamma,
            "covariance_G_U": cov_gu, "covariance_G_M": cov_gm, "covariance_U_M": cov_um}


def calibrate_common_K(development_parameters: Iterable[KineticParameters]) -> dict[str, Any]:
    """Propose one positive K from analytic no-feedback developmental references.

    The rule is the median of stationary mean M across all development genes,
    with feedback removed analytically.  It never uses a blind-network endpoint.
    The returned artifact records inputs and a hash; freezing it is a separate
    scientific decision, and this helper does not alter any parameter object.
    """
    parameters = list(development_parameters)
    if not parameters:
        raise ValueError("at least one development parameter set is required")
    references = []
    inputs = []
    for params in parameters:
        # At M=0 every regulatory signal is zero, so production is exactly g(b).
        s = np.asarray(params.production(np.zeros((1, params.n_genes)))).reshape(-1)
        p = np.asarray(params.kon) / (np.asarray(params.kon) + np.asarray(params.koff))
        means = s * p / np.asarray(params.gamma)
        if not np.isfinite(means).all() or np.any(means <= 0):
            raise ValueError("no-feedback reference means must be finite and positive")
        references.append(means.tolist())
        inputs.append(params.to_dict())
    payload = {"reference": "analytic_no_feedback_mean_M", "reference_means": references,
               "development_inputs": inputs, "rule": "median_all_development_genes"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"schema_version": "scmultisim-common-K-v1", "K": float(np.median(np.concatenate(references))),
            "reference": payload["reference"], "reference_means": references,
            "rule": payload["rule"], "sample_sha256": digest,
            "development_network_count": len(parameters),
            "status": "development_candidate_not_automatically_frozen"}


def calibrate_tau(
    parameters: KineticParameters,
    initial: CellState,
    seed: int,
    *,
    tau: float = 1 / 32,
    levels: int = 3,
    times: Iterable[float] = (0.25, 1.0, 4.0),
    target: int | None = None,
    dose: float = 0.0,
    tolerances: StationarityConfig | None = None,
) -> dict[str, Any]:
    """Compare tau, tau/2, ... using identical initial states and promoter streams.

    Marginal and joint distribution discrepancies are reported separately from
    paired numerical errors; these paired paths are generator diagnostics only.
    Passing one case does not freeze tau for all development/blind networks.
    """
    _positive_finite(tau, "tau")
    _positive_int(levels, "levels", minimum=2)
    samples = tuple(float(t) for t in times)
    if not samples or any(not math.isfinite(t) or t <= 0 for t in samples) or any(b <= a for a, b in zip(samples, samples[1:])):
        raise ValueError("times must be finite, positive, and strictly increasing")
    config = StationarityConfig() if tolerances is None else tolerances
    snapshots: list[list[CellState]] = []
    intervals = [tau / 2**index for index in range(levels)]
    for interval in intervals:
        simulator = Simulator(parameters, initial.copy(), seed=seed, tau=interval, target=target, dose=dose)
        snapshots.append([simulator.advance(t) for t in samples])
    comparisons = []
    for level in range(1, levels):
        checks = []
        for t, coarse, fine in zip(samples, snapshots[level - 1], snapshots[level], strict=True):
            diagnostics = compare_distributions(coarse, fine, **_comparison_kwargs(config))
            delta = _state_array(coarse) - _state_array(fine)
            checks.append({"time": t, "distribution": diagnostics,
                           "paired_max_absolute_error": float(np.max(np.abs(delta))),
                           "paired_root_mean_square_error": float(np.sqrt(np.mean(delta**2))),
                           "promoter_states_identical": bool(np.array_equal(coarse.G, fine.G))})
        passed = all(item["distribution"]["passed"] and item["promoter_states_identical"] for item in checks)
        comparisons.append({"coarse_tau": intervals[level - 1], "fine_tau": intervals[level],
                            "passed": passed, "times": checks})
    return {"schema_version": "scmultisim-tau-convergence-v1", "seed": int(seed),
            "initial_cells": int(initial.M.shape[0]), "comparisons": comparisons,
            "finest_comparison_passed": comparisons[-1]["passed"],
            "finest_tau_candidate": intervals[-1] if comparisons[-1]["passed"] else None,
            "status": "development_diagnostic_not_automatically_frozen"}
