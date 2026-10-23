"""Tiny kernel and filesystem fixtures; not scientific calibration evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from grn_experiment.protocol import COMMON_BASELINE, Condition, SteadyKey, make_split
from grn_experiment.scmultisim import development_parameters, parameter_digest
from grn_experiment.scmultisim_observations import SnapshotSampler, key_id, stream_seed
from grn_experiment.generate_scmultisim import planned_conditions


def sampler(seed=4):
    # An explicit short-burnin unit fixture, not a stationarity certificate.
    params = development_parameters(7, n_genes=4)
    report = {"parameter_sha256": parameter_digest(params), "tau": .125, "target": None, "dose": 0.}
    return SnapshotSampler(params, seed=seed, tau=0.125,
                           baseline_stationarity=SimpleNamespace(passed=True, burnin=0.25, report=report),
                           cells_per_replicate=3)


def test_snapshot_is_order_independent_and_controls_use_distinct_lineages():
    generator = sampler()
    condition = Condition(0, 0.7, 0.25)
    first = generator.sample_with_truth(condition)
    generator.sample_condition(Condition(0, 0.3, 0.5))
    again = generator.sample_with_truth(condition)
    np.testing.assert_array_equal(first.snapshot.treated, again.snapshot.treated)
    np.testing.assert_array_equal(first.snapshot.matched_control, again.snapshot.matched_control)
    streams = first.provenance["streams"]
    assert len({stream["initial_seed"] for stream in streams}) == 6
    assert len({stream["trajectory_seed"] for stream in streams}) == 6
    assert all(stream["initial_seed"] != stream["trajectory_seed"] for stream in streams)
    assert first.snapshot.treated.shape == (3, 3, 4)
    np.testing.assert_array_equal(first.snapshot.treated, np.floor(first.truth["treated_final_M"]))
    assert first.truth["treated_production_jacobian"].shape == (3, 3, 4, 4)
    np.testing.assert_array_equal(first.truth["treated_production_jacobian"],
                                  development_parameters(7, n_genes=4).production_jacobian(first.truth["treated_final_M"]))
    assert "G" not in vars(first.snapshot) and "rho" not in vars(first.snapshot)


def test_omitting_quadratic_truth_keeps_identical_counts_and_reconstructible_jacobians():
    generator = sampler()
    condition = Condition(0, 0.7, 0.25)
    full = generator.sample_with_truth(condition)
    generator.save_production_jacobians = False
    compact = generator.sample_with_truth(condition)
    assert compact.provenance["production_jacobians_saved"] is False
    for arm, observed in (("treated", "treated"), ("control", "matched_control")):
        np.testing.assert_array_equal(getattr(full.snapshot, observed), getattr(compact.snapshot, observed))
        assert f"{arm}_production_jacobian" not in compact.truth
        for species in ("G", "U", "M"):
            for stage in ("initial", "final"):
                name = f"{arm}_{stage}_{species}"
                np.testing.assert_array_equal(full.truth[name], compact.truth[name])
        reconstructed = development_parameters(7, n_genes=4).production_jacobian(compact.truth[f"{arm}_final_M"])
        np.testing.assert_array_equal(reconstructed, full.truth[f"{arm}_production_jacobian"])


def test_sampler_rejects_nonboolean_truth_option():
    params = development_parameters(7, n_genes=4)
    report = {"parameter_sha256": parameter_digest(params), "tau": .125, "target": None, "dose": 0.}
    with pytest.raises(ValueError, match="save_production_jacobians"):
        SnapshotSampler(params, seed=4, tau=.125,
                        baseline_stationarity=SimpleNamespace(passed=True, burnin=.25, report=report),
                        save_production_jacobians="false")


def test_common_baseline_is_one_unperturbed_arm_with_unchanged_hidden_state():
    result = sampler().sample_with_truth(COMMON_BASELINE)
    assert result.snapshot.matched_control is None
    assert len(result.provenance["streams"]) == 3
    for species in ("G", "U", "M"):
        np.testing.assert_array_equal(result.truth[f"treated_initial_{species}"], result.truth[f"treated_final_{species}"])


def test_timepoints_and_strengths_do_not_reuse_initial_cells():
    generator = sampler()
    first = generator.sample_with_truth(Condition(0, 0.3, 0.25))
    later = generator.sample_with_truth(Condition(0, 0.3, 0.5))
    assert first.provenance["streams"][0]["initial_seed"] != later.provenance["streams"][0]["initial_seed"]
    assert key_id(Condition(0, 0.3, 0.25)) != key_id(SteadyKey(0, 0.3))


def test_failed_stationarity_cannot_be_used_as_initial_or_final_distribution():
    params = development_parameters(7, n_genes=4)
    failed = SimpleNamespace(passed=False, burnin=None)
    with pytest.raises(ValueError, match="Baseline stationarity"):
        SnapshotSampler(params, seed=0, tau=0.125, baseline_stationarity=failed)
    with pytest.raises(ValueError, match="condition-specific"):
        sampler().sample_with_truth(SteadyKey(0, 0.7), stationary_result=failed)


def test_protocol_grid_counts_and_seed_validation():
    split = make_split(tuple(range(16)), 17)
    assert len(planned_conditions(split, "initial")) == 37
    assert len(planned_conditions(split, "full")) == 85
    assert len(planned_conditions(split, "pool")) == 163
    assert planned_conditions(split, "full").count(COMMON_BASELINE) == 1
    with pytest.raises(ValueError):
        stream_seed(-1, "condition")
    with pytest.raises(ValueError):
        planned_conditions(split, "blind")


def test_stationarity_certificate_cannot_be_reused_for_different_network():
    params = development_parameters(8, n_genes=4)
    wrong = SimpleNamespace(passed=True, burnin=.25,
                            report={"parameter_sha256": parameter_digest(development_parameters(7, n_genes=4)),
                                    "tau": .125, "target": None, "dose": 0.})
    with pytest.raises(ValueError, match="does not match"):
        SnapshotSampler(params, seed=0, tau=.125, baseline_stationarity=wrong)


def test_stationary_outcome_requires_exact_intervention_diagnostic():
    params = development_parameters(7, n_genes=4)
    wrong = SimpleNamespace(passed=True, burnin=.25,
                            report={"parameter_sha256": parameter_digest(params),
                                    "tau": .125, "target": 0, "dose": .3})
    with pytest.raises(ValueError, match="does not match"):
        sampler().sample_with_truth(SteadyKey(0, .7), stationary_result=wrong)
