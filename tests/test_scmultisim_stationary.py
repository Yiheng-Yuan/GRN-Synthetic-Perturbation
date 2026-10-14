"""Stationarity diagnostics test their claims, not a universal convergence claim."""

import json

import numpy as np
import pytest

from grn_experiment.scmultisim import CellState, KineticParameters
import grn_experiment.scmultisim_stationary as stationary


def parameters(A=None):
    n = 1 if A is None else len(A)
    return KineticParameters(
        A=np.zeros((n, n)) if A is None else np.asarray(A),
        b=np.zeros(n), kon=np.ones(n), koff=np.ones(n),
        beta=np.full(n, 2.0), gamma=np.ones(n), rho=np.full(n, 0.8),
        regulators=tuple(range(n)), K=20.0,
    )


@pytest.mark.parametrize("kwargs", [
    {"min_time": -1}, {"max_time": float("nan")}, {"window": 0},
    {"consecutive_windows": True}, {"diagnostic_cells": 1},
    {"diagnostic_cells": 2.5}, {"samples_per_window": 0},
    {"distribution_tol": -1}, {"atol": 0},
    {"min_time": 10, "max_time": 11, "window": 1},
])
def test_invalid_stationarity_rules_rejected(kwargs):
    with pytest.raises(ValueError):
        stationary.StationarityConfig(**kwargs)


def test_joint_distance_detects_different_dependence_with_equal_marginals():
    a = np.tile([[-1.0, -1.0], [1.0, 1.0]], (40, 1))
    b = np.tile([[-1.0, 1.0], [1.0, -1.0]], (40, 1))
    result = stationary.compare_distributions(a, b)
    assert result["max_standardized_marginal_wasserstein"] == 0
    assert result["max_standardized_projected_wasserstein"] > 0.2
    assert result["passed"] is False


def test_distribution_distance_handles_unequal_sample_sizes():
    a = np.array([[0.0], [1.0]])
    b = np.array([[0.0], [0.0], [1.0], [1.0]])
    assert stationary.compare_distributions(a, b, variance_rtol=1.0)["max_standardized_marginal_wasserstein"] == 0
    with pytest.raises(ValueError, match="finite"):
        stationary.compare_distributions(np.array([[np.nan], [1.0]]), a)


def test_analytic_fluid_moments_and_sustained_dose():
    params = parameters()
    s = params.production(np.zeros((1, 1)))[0, 0]
    moment = stationary.analytic_no_feedback_moments(params)
    assert moment["mean_G"] == pytest.approx([0.5])
    assert moment["mean_U"] == pytest.approx([s / 4])
    assert moment["mean_M"] == pytest.approx([s / 2])
    assert moment["variance_U"] == pytest.approx([s**2 / 32])
    assert moment["variance_M"] == pytest.approx([5 * s**2 / 72])
    pert = stationary.analytic_no_feedback_moments(params, target=0, dose=0.5)
    factor = 1 - 0.8 * 0.5
    assert pert["mean_M"] == pytest.approx(moment["mean_M"] * factor)
    assert pert["variance_M"] == pytest.approx(moment["variance_M"] * factor**2)
    with pytest.raises(ValueError, match="no-feedback"):
        stationary.analytic_no_feedback_moments(parameters([[0.0, 0.1], [0.0, 0.0]]))


def test_common_K_is_derived_from_no_feedback_not_the_feedback_endpoint():
    base = parameters()
    feedback = parameters([[0.0, 2.0], [0.0, 0.0]])
    first = stationary.calibrate_common_K([base])
    second = stationary.calibrate_common_K([feedback])
    assert first["K"] == second["K"]
    assert first["K"] == pytest.approx(stationary.analytic_no_feedback_moments(base)["mean_M"][0])
    assert first["sample_sha256"] == stationary.calibrate_common_K([base])["sample_sha256"]
    assert first["sample_sha256"] != second["sample_sha256"]
    assert first["status"] == "development_candidate_not_automatically_frozen"
    json.dumps(first, allow_nan=False)
    with pytest.raises(ValueError):
        stationary.calibrate_common_K([])


def test_stationarity_uses_contrasting_joint_starts_and_full_windows(monkeypatch):
    seen = []

    class SettledSimulator:
        def __init__(self, params, initial, seed, tau, target, dose):
            seen.append((initial.copy(), seed, target, dose))
            self.shape = initial.M.shape

        def advance(self, time):
            return CellState(G=np.ones(self.shape, dtype=np.int8),
                             U=np.full(self.shape, 2.0), M=np.full(self.shape, 3.0))

    monkeypatch.setattr(stationary, "Simulator", SettledSimulator)
    config = stationary.StationarityConfig(min_time=2, max_time=10, window=1,
                                           consecutive_windows=2, diagnostic_cells=8,
                                           samples_per_window=3)
    result = stationary.assess_stationarity(parameters(), seed=2, tau=0.25, config=config, target=0, dose=0.7)
    assert result.passed
    # One reference window, then two successful consecutive comparisons.
    assert result.burnin == 5
    assert len(result.report["windows"]) == 3
    assert result.report["windows"][0]["passed"] is False
    assert result.report["windows"][1]["between_starts"]["left_rows"] == 24
    assert result.report["windows"][1]["between_starts"]["features"] == 3
    assert not np.any(seen[0][0].G)
    assert not np.any(seen[0][0].U)
    assert np.all(seen[1][0].G == 1)
    assert np.all(seen[1][0].M > 0)
    assert seen[0][1] != seen[1][1]
    assert seen[0][2:] == (0, 0.7)
    assert result.report["within_lineage_window_samples_are_correlated"]
    json.dumps(result.to_dict(), allow_nan=False)


def test_exhausted_budget_never_returns_stationary_burnin():
    config = stationary.StationarityConfig(min_time=0, max_time=2, window=1,
                                           consecutive_windows=3, diagnostic_cells=8,
                                           samples_per_window=1)
    result = stationary.assess_stationarity(parameters(), seed=22, tau=0.25, config=config)
    assert not result.passed
    assert result.burnin is None
    assert result.elapsed_time == 2
    assert result.report["reason"] == "budget_exhausted"
    assert result.report["status"] == "stationarity_not_established"


def test_fresh_lineages_reproducible_and_match_no_edge_moments():
    params = parameters()
    actual = stationary.sample_initial_states(params, 1024, seed=2032, tau=0.25, burnin=24)
    again = stationary.sample_initial_states(params, 1024, seed=2032, tau=0.25, burnin=24)
    assert np.array_equal(actual.G, again.G)
    assert np.array_equal(actual.U, again.U)
    assert np.array_equal(actual.M, again.M)
    expected = stationary.analytic_no_feedback_moments(params)
    for name in ("G", "U", "M"):
        values = getattr(actual, name)
        assert values.mean(axis=0) == pytest.approx(expected[f"mean_{name}"], rel=0.12, abs=0.04)
        assert values.var(axis=0) == pytest.approx(expected[f"variance_{name}"], rel=0.22, abs=0.04)
    with pytest.raises(ValueError):
        stationary.sample_initial_states(params, 4, seed=3, tau=0.25, burnin=0)


def test_no_feedback_tau_halving_agrees_and_does_not_freeze_rule():
    params = parameters()
    initial = CellState(G=np.zeros((8, 1), dtype=np.int8), U=np.zeros((8, 1)), M=np.zeros((8, 1)))
    result = stationary.calibrate_tau(params, initial, seed=33, tau=0.125, levels=2, times=(0.25, 1.0))
    assert result["finest_comparison_passed"]
    assert result["finest_tau_candidate"] == 0.0625
    for item in result["comparisons"][0]["times"]:
        assert item["promoter_states_identical"]
        assert item["paired_max_absolute_error"] < 1e-10
    assert result["status"] == "development_diagnostic_not_automatically_frozen"
    with pytest.raises(ValueError):
        stationary.calibrate_tau(params, initial, seed=33, times=(1.0, 0.25))
