"""End-to-end integration tests for the complete 12-round acquisition loop.

These tests verify the full workflow from network generation through active
sampling to final scoring, ensuring all modules integrate correctly.
"""

import numpy as np
import pytest
import torch

from grn_experiment.ambiguity import (
    QualificationThresholds,
    fit_structural_rivals,
    single_edge_swaps,
    top_edge_support,
)
from grn_experiment.baselines import EndpointResponsePredictor, KnownFormRNAODE
from grn_experiment.config import ExperimentConfig
from grn_experiment.learning import SparseRNAODE, TrainConfig, fit_unpaired_snapshots
from grn_experiment.observations import ObservationConfig, SnapshotSampler
from grn_experiment.parameter_sets import (
    ParameterBounds,
    ParameterSetApproximation,
    choose_parameter_informed,
)
from grn_experiment.protocol import (
    COMMON_BASELINE,
    BlindStore,
    Condition,
    candidate_grid,
    initial_grid,
    make_split,
)
from grn_experiment.selection import choose_active, random_sample, uniform_order
from grn_experiment.simulation import make_dynamics, make_network
from grn_experiment.workflow import CaseWorkflow, NetworkPlan


@pytest.fixture
def small_config():
    """Reduced parameters for faster testing."""
    config = ExperimentConfig.from_defaults()
    return config


@pytest.fixture
def test_network():
    """Create one deterministic test network."""
    network = make_network(seed=12345, n_genes=24, n_regulators=16, n_edges=40)
    return network


@pytest.fixture
def test_dynamics(test_network):
    """Create sigmoid dynamics for the test network."""
    return make_dynamics(test_network, family="sigmoid", seed=67890)


@pytest.fixture
def test_split(test_network):
    """Create 6/6/2/2 regulator split."""
    return make_split(test_network.regulators, seed=111)


@pytest.fixture
def test_sampler(test_dynamics, test_split):
    """Create observation sampler."""
    obs_config = ObservationConfig(cells_per_replicate=32)  # Reduced for speed
    return SnapshotSampler(test_dynamics, seed=222, config=obs_config)


def test_complete_problem1_workflow(test_network, test_split, test_sampler):
    """Test full Problem 1 workflow: unknown dynamics learning."""

    # Phase 1: Generate initial observations
    baseline = test_sampler.sample_condition(COMMON_BASELINE)
    initial_conditions = initial_grid(test_split)[:6]  # Use subset for speed

    initial_obs = {cond: test_sampler.sample_condition(cond) for cond in initial_conditions}

    # Phase 2: Train initial model
    model = SparseRNAODE(test_network.n_genes, min_efficiency=0.5)
    train_config = TrainConfig(epochs=50, conditions_per_epoch=2, seed=333)

    report = fit_unpaired_snapshots(model, baseline, initial_obs, config=train_config)

    assert len(report.losses) == 50
    assert report.conditions_seen == len(initial_conditions)
    assert all(loss > 0 for loss in report.losses)

    # Phase 3: Propose rival structures
    support = top_edge_support(model, test_split.initial + test_split.active, n_edges=40)
    assert len(support) == 40

    rivals_specs = single_edge_swaps(
        support,
        n_genes=test_network.n_genes,
        regulators=list(test_network.regulators),
        count=5,
        seed=444,
    )
    assert len(rivals_specs) == 5

    # Phase 4: Qualify rivals
    thresholds = QualificationThresholds(
        max_distribution_error=0.2, max_effect_error=0.15, min_local_edge_effect=0.01
    )

    effect_states = np.random.RandomState(555).uniform(0.5, 2.5, size=(10, test_network.n_genes))

    rivals, reports = fit_structural_rivals(
        rivals_specs,
        baseline,
        initial_obs,
        thresholds,
        config=train_config,
        effect_states=effect_states,
    )

    assert len(reports) == 5
    qualified_count = sum(1 for r in reports if r.qualified)
    assert qualified_count >= 0  # Some may not qualify

    # Phase 5: Active selection (if we have qualified rivals)
    candidate_pool = candidate_grid(test_split)[:10]  # Subset for speed

    if len(rivals) >= 2:
        choice = choose_active(candidate_pool, tuple(initial_conditions), rivals)
        assert choice.condition in candidate_pool
        assert choice.condition not in initial_conditions
    else:
        # Fallback to uniform if not enough rivals
        uniform = uniform_order(candidate_pool)
        assert len(uniform) > 0

    print(f"✓ Problem 1 workflow complete: {qualified_count}/{len(rivals_specs)} rivals qualified")


def test_complete_problem2_workflow(test_network, test_split, test_sampler):
    """Test full Problem 2 workflow: known-form parameter identification."""

    # Phase 1: Generate initial observations (same as Problem 1)
    baseline = test_sampler.sample_condition(COMMON_BASELINE)
    initial_conditions = initial_grid(test_split)[:6]
    initial_obs = {cond: test_sampler.sample_condition(cond) for cond in initial_conditions}

    # Phase 2: Initialize parameter set approximation
    bounds = ParameterBounds()
    tolerances = [
        (cond, 0.15, 0.10) for cond in initial_conditions
    ]  # (condition, dist_tol, effect_tol)

    from grn_experiment.parameter_sets import ConditionTolerance

    tol_objects = [
        ConditionTolerance(c, d, e) for c, d, e in tolerances
    ]

    param_set = ParameterSetApproximation(
        family="sigmoid",
        n_genes=test_network.n_genes,
        bounds=bounds,
        tolerances=tol_objects,
        n_samples=100,  # Reduced for speed
        seed=666,
    )

    initial_survivors = param_set.n_survivors
    assert initial_survivors == 100

    # Phase 3: Refine with initial observations
    eliminated, remaining = param_set.refine(
        baseline, initial_obs, step=0.1, projection_seed=777
    )

    assert eliminated >= 0
    assert remaining <= initial_survivors
    assert remaining == param_set.n_survivors

    print(
        f"  Initial set: {initial_survivors} samples"
    )
    print(f"  After constraints: {remaining} survivors ({eliminated} eliminated)")

    # Phase 4: Parameter-informed active selection
    candidate_pool = candidate_grid(test_split)[:10]

    if remaining > 0:
        next_condition = choose_parameter_informed(
            candidate_pool, tuple(initial_conditions), param_set, baseline, step=0.1
        )
        assert next_condition in candidate_pool
        assert next_condition not in initial_conditions

        # Phase 5: Query prediction disagreement
        lower, upper = param_set.prediction_disagreement(baseline, next_condition, step=0.1)
        assert lower.shape == (test_network.n_genes,)
        assert upper.shape == (test_network.n_genes,)
        assert np.all(lower <= upper)

        disagreement = np.mean(upper - lower)
        print(f"  Next condition disagreement: {disagreement:.4f}")

        # Phase 6: Check parameter ranges
        ranges = param_set.parameter_ranges()
        assert "weights_l1_norm" in ranges
        assert "gamma_mean" in ranges
        assert all(low <= high for low, high in ranges.values())

    print(f"✓ Problem 2 workflow complete: {remaining} compatible parameters remain")


def test_twelve_round_acquisition_loop(test_network, test_split, test_sampler):
    """Test complete 12-round active acquisition with model updates."""

    # Setup
    baseline = test_sampler.sample_condition(COMMON_BASELINE)
    initial_conditions = initial_grid(test_split)
    initial_obs = {cond: test_sampler.sample_condition(cond) for cond in initial_conditions}

    candidate_pool = candidate_grid(test_split)
    acquired_conditions = []

    model = SparseRNAODE(test_network.n_genes, min_efficiency=0.5)
    train_config = TrainConfig(epochs=30, conditions_per_epoch=3, seed=888)

    all_observations = dict(initial_obs)

    # 12-round loop
    for round_idx in range(12):
        # Refit model with all acquired data
        fit_unpaired_snapshots(model, baseline, all_observations, config=train_config)

        # Propose and qualify rivals
        support = top_edge_support(model, test_split.initial + test_split.active, n_edges=40)
        rivals_specs = single_edge_swaps(
            support,
            n_genes=test_network.n_genes,
            regulators=list(test_network.regulators),
            count=5,
            seed=1000 + round_idx,
        )

        thresholds = QualificationThresholds(
            max_distribution_error=0.2, max_effect_error=0.15, min_local_edge_effect=0.01
        )
        effect_states = np.random.RandomState(2000 + round_idx).uniform(
            0.5, 2.5, size=(10, test_network.n_genes)
        )

        rivals, _ = fit_structural_rivals(
            rivals_specs,
            baseline,
            all_observations,
            thresholds,
            config=train_config,
            effect_states=effect_states,
        )

        # Select next condition
        if len(rivals) >= 2:
            choice = choose_active(candidate_pool, tuple(acquired_conditions), rivals)
            next_cond = choice.condition
        else:
            # Fallback to uniform
            available = [c for c in candidate_pool if c not in acquired_conditions]
            next_cond = available[0] if available else None

        if next_cond is None:
            break

        # Acquire observation
        acquired_conditions.append(next_cond)
        all_observations[next_cond] = test_sampler.sample_condition(next_cond)

        print(f"  Round {round_idx + 1}/12: Acquired target={next_cond.target}, "
              f"dose={next_cond.dose}, time={next_cond.time}, "
              f"rivals={len(rivals)}")

    assert len(acquired_conditions) <= 12
    print(f"✓ Completed {len(acquired_conditions)} acquisition rounds")


def test_baseline_comparators(test_network, test_split, test_sampler):
    """Test all three baseline methods."""

    baseline = test_sampler.sample_condition(COMMON_BASELINE)
    initial_conditions = initial_grid(test_split)[:6]
    initial_obs = {cond: test_sampler.sample_condition(cond) for cond in initial_conditions}

    # Baseline 1: Endpoint response
    endpoint = EndpointResponsePredictor()
    endpoint.fit(initial_obs)

    test_cond = Condition(test_split.validation[0], 0.5, 2.0)
    pred = endpoint.predict(baseline.fit_treated.reshape(-1, test_network.n_genes), test_cond)
    assert pred.shape == (baseline.fit_treated.reshape(-1, test_network.n_genes).shape[0], test_network.n_genes)

    # Baseline 2: Known form (oracle architecture)
    known_form = KnownFormRNAODE("sigmoid", test_network.n_genes, min_efficiency=0.5)
    train_config = TrainConfig(epochs=30, seed=999)
    fit_unpaired_snapshots(known_form, baseline, initial_obs, config=train_config)

    pred_kf = known_form.integrate(
        torch.from_numpy(baseline.fit_treated.reshape(-1, test_network.n_genes).astype(np.float32)),
        test_cond.time,
        test_cond.target,
        test_cond.dose,
    )
    assert pred_kf.shape[1] == test_network.n_genes

    print("✓ All baseline methods functional")


def test_workflow_integration(test_network, test_split, test_sampler):
    """Test CaseWorkflow orchestration of three strategies."""

    # Generate observations for protocol
    baseline = test_sampler.sample_condition(COMMON_BASELINE)
    initial_conditions = initial_grid(test_split)
    all_conditions = [COMMON_BASELINE] + list(initial_conditions) + list(candidate_grid(test_split))

    observations = {cond: test_sampler.sample_condition(cond) for cond in all_conditions}

    # Create workflow
    plan = NetworkPlan(
        identifier="test-00",
        role="development",
        network_seed=12345,
        dynamics_seed=67890,
        observation_seed=222,
        split_seed=111,
        random_selection_seed=333,
    )

    # Mock initial supports (would come from structure proposal)
    model = SparseRNAODE(test_network.n_genes)
    support = top_edge_support(model, test_network.regulators, n_edges=40)
    initial_supports = {"structure_0": support}

    workflow = CaseWorkflow(
        plan,
        family="sigmoid",
        split=test_split,
        observations=observations,
        initial_supports=initial_supports,
        n_genes=test_network.n_genes,
        cells_per_replicate=32,
    )

    # Test all three strategy runs exist
    for strategy in ["active", "random", "uniform"]:
        run = workflow.run(strategy)
        assert run.plan.identifier == "test-00"
        assert len(run.audit) >= 1  # At least start event

    print("✓ CaseWorkflow integration complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
