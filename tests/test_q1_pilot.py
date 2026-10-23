"""Tests for the small unknown-form development learner, not science claims."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grn_experiment.protocol import Condition, ReplicatedSnapshot
from grn_experiment.q1_pilot import (LatentPerturbationODE, PilotConfig,
                                    distribution_loss, fit_pilot, predict_population)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_observations():
    rng = np.random.default_rng(5)
    baseline = rng.poisson(5, (16, 4)).astype(float)
    observations = {}
    for time in (0.25, 4.0):
        for target in (0, 1):
            control = rng.poisson(5, (3, 8, 4)).astype(float)
            treated = control.copy()
            treated[:, :, target] = np.maximum(0, treated[:, :, target] - time * 0.7)
            observations[Condition(target, 0.7, time)] = ReplicatedSnapshot(treated, control)
    return baseline, observations


def test_initial_decode_and_zero_time_do_not_depend_on_future_target():
    model = LatentPerturbationODE(4, hidden=8)
    counts = torch.tensor([[0., 1., 2., 9.], [3., 4., 0., 2.]])
    initial = model.initial_state(counts, torch.zeros_like(counts))
    torch.testing.assert_close(model.decode(initial), counts)
    for target in (0, 2):
        mask = torch.zeros_like(counts)
        mask[:, target] = 0.7
        torch.testing.assert_close(model.integrate(initial, mask, 0), initial)


def test_field_is_generic_local_input_is_shared_and_zero_mask_disables_it():
    model = LatentPerturbationODE(4, hidden=8)
    state = torch.randn(2, 4, 2)
    zeros = torch.zeros(2, 4)
    expected = model.core(state.flatten(-2)).reshape_as(state)
    torch.testing.assert_close(model.vector_field(state, zeros), expected)
    assert not any(isinstance(module, torch.nn.Embedding) for module in model.modules())
    assert not hasattr(model, "gamma") and not hasattr(model, "rho") and not hasattr(model, "A")


def test_population_loss_has_no_row_correspondence():
    torch.manual_seed(2)
    a, b = torch.rand(2, 12, 4), torch.rand(2, 12, 4)
    directions = torch.randn(4, 8)
    original = distribution_loss(a, b, directions)
    permuted = distribution_loss(a[:, torch.randperm(12)], b[:, torch.randperm(12)], directions)
    torch.testing.assert_close(original, permuted)


def test_fitting_ignores_third_repeat_and_is_reproducible():
    baseline, observations = fixture_observations()
    config = PilotConfig(updates=4, cells=8, hidden=8, projections=4, seed=2)
    first, history = fit_pilot(baseline, observations, config=config)
    poisoned = {}
    for condition, snapshot in observations.items():
        a, b = snapshot.treated.copy(), snapshot.matched_control.copy()
        a[2], b[2] = 9999, 1000
        poisoned[condition] = ReplicatedSnapshot(a, b)
    second, other = fit_pilot(baseline, poisoned, config=config)
    assert history == other
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    assert all(np.isfinite(row["loss"]) for row in history)
    prediction, control = predict_population(first, baseline, Condition(0, .7, 1.), cells=16, seed=3)
    assert prediction.shape == control.shape == (16, 4)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    repeated = predict_population(first, baseline, Condition(0, .7, 1.), cells=16, seed=3)
    np.testing.assert_array_equal(prediction, repeated[0])


def test_held_time_is_rejected_by_training_interface():
    baseline, observations = fixture_observations()
    observations[Condition(0, .7, 1.)] = next(iter(observations.values()))
    with pytest.raises(ValueError, match="only targeted"):
        fit_pilot(baseline, observations, config=PilotConfig(updates=2))


@pytest.mark.parametrize("change", [{"updates": 0}, {"seed": -1}, {"step": .5},
                                     {"learning_rate": float("nan")}, {"cells": 1}])
def test_bad_training_config_rejected(change):
    with pytest.raises(ValueError):
        PilotConfig(**change)


def test_import_does_not_load_generator_modules():
    source = Path(__file__).parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}
    script = ("import grn_experiment.q1_pilot, sys; "
              "assert not any(k.startswith('grn_experiment.scmultisim') or "
              "k == 'grn_experiment.simulation' for k in sys.modules)")
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_fullcourse_times_are_balanced_and_extra_target_weight_can_be_zero():
    baseline, original = fixture_observations()
    times = (0.25, 1.0, 2.0, 4.0, 8.0, 48.0)
    observations = {Condition(q, .7, t): original[Condition(q, .7, .25)]
                    for q in (0, 1) for t in times}
    config = PilotConfig(updates=6, cells=4, hidden=8, projections=4,
                         fit_times=times, target_weight=0, seed=3)
    logged = []
    model, history = fit_pilot(baseline, observations, config=config, on_update=logged.append)
    assert tuple(row["time"] for row in history) == times
    assert logged == history
    for row in history:
        assert row["loss"] == pytest.approx(row["distribution"] + row["control"] + row["effect"])
    a, b = predict_population(model, baseline, Condition(2, .7, 48), cells=4, step=.125)
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_training_requires_a_complete_common_time_grid():
    baseline, observations = fixture_observations()
    observations.pop(Condition(1, .7, 4))
    with pytest.raises(ValueError, match="same target/dose"):
        fit_pilot(baseline, observations, config=PilotConfig(updates=2))


@pytest.mark.parametrize("change", [
    {"fit_times": ()}, {"fit_times": (1, .25)}, {"fit_times": (.25, .25)},
    {"fit_times": (0, 1)}, {"fit_times": (True, 4)},
    {"fit_times": (float("nan"),)}, {"target_weight": -1},
    {"effect_weight": float("inf")},
])
def test_invalid_fullcourse_config_rejected(change):
    with pytest.raises(ValueError):
        PilotConfig(**change)


def test_held_target_poison_does_not_change_fitted_weights():
    from grn_experiment.q1_pilot_data import OpenedInitialData, split_targets
    rng = np.random.default_rng(7)
    baseline = rng.poisson(5, (3, 8, 6))
    observations = {}
    for q in range(6):
        for t in (.25, 4):
            counts = rng.poisson(5, (3, 8, 6))
            observations[Condition(q, .7, t)] = ReplicatedSnapshot(counts, counts.copy())
    data = OpenedInitialData(baseline, observations, {"initial_targets": list(range(6))})
    fitting, held = split_targets(data)
    poisoned = {**observations, **{key: ReplicatedSnapshot(
        np.full_like(value.treated, 9999), np.full_like(value.matched_control, 2000))
        for key, value in held.items()}}
    altered = OpenedInitialData(baseline, poisoned, data.provenance)
    other_fitting, _ = split_targets(altered)
    config = PilotConfig(updates=2, hidden=8, cells=4, projections=4, target_weight=0)
    first, first_history = fit_pilot(baseline[:2].reshape(-1, 6), fitting, config=config)
    second, second_history = fit_pilot(baseline[:2].reshape(-1, 6), other_fitting, config=config)
    assert first_history == second_history
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)


def test_fullcourse_cli_freezes_split_before_fit_and_writes_valid_artifacts(tmp_path, monkeypatch):
    """Exercise orchestration with one real update, not long-time numerics."""
    from grn_experiment import q1_pilot as pilot
    from grn_experiment import q1_pilot_data as loading

    baseline = np.full((3, 4, 6), 20, dtype=np.int64)
    baseline[2] = 9000  # The third baseline repeat must never enter fitting.
    observations = {}
    for target in range(6):
        for duration in (.25, 4.):
            control = np.full((3, 4, 6), 20, dtype=np.int64)
            treated = control.copy()
            treated[..., target] -= 1
            observations[Condition(target, .7, duration)] = ReplicatedSnapshot(treated, control)
    provenance = {"initial_targets": list(range(6)), "phase": "development"}
    data = loading.OpenedInitialData(baseline, observations, provenance)
    fitting, held = loading.split_targets(data)
    # A conspicuous future response catches accidentally forwarding held data.
    observations.update({key: ReplicatedSnapshot(np.full_like(value.treated, 9999),
                                                 np.full_like(value.matched_control, 1000))
                         for key, value in held.items()})
    monkeypatch.setattr(loading, "load_fullcourse_initial", lambda *_: data)
    prepare_args = SimpleNamespace(fullcourse=True, archive=tmp_path / "unused.tar.gz",
                                   prefix="unused/", updates=None, seed=20261022,
                                   split_seed=20261022)
    assert pilot._prepare_experiment(prepare_args)[3].updates == 600

    output = tmp_path / "run"
    real_fit, real_evaluate = pilot.fit_pilot, pilot._evaluate
    seen = []

    def guarded_fit(fit_baseline, fit_observations, **kwargs):
        seen.append("fit")
        registered = json.loads((output / "run_plan.json").read_text())["source_snapshot"]
        assert set(registered["files"]) == {
            "q1_pilot.py", "q1_pilot_data.py", "q1_pilot_evaluation.py", "protocol.py", "__init__.py"}
        for name, checksum in registered["files"].items():
            copied = output / registered["directory"] / name
            assert copied.read_bytes() == Path(pilot.__file__).with_name(name).read_bytes()
            assert hashlib.sha256(copied.read_bytes()).hexdigest() == checksum
        np.testing.assert_array_equal(fit_baseline, baseline[:2].reshape(-1, 6))
        assert set(fit_observations) == set(fitting)
        assert not set(fit_observations) & set(held)
        assert all(value.fit_treated.max() < 9999 for value in fit_observations.values())
        return real_fit(fit_baseline, fit_observations, **kwargs)

    def guarded_evaluate(model, *args):
        seen.append("evaluate")
        checkpoint = torch.load(output / "model.pt", weights_only=True)
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(checkpoint["state_dict"][name], weight, rtol=0, atol=0)
        return real_evaluate(model, *args)

    def cheap_prediction(model, fit_baseline, condition, **kwargs):
        # The integrator has separate unit tests; this test checks file/data flow.
        assert condition.target in range(model.n_genes)
        population = np.asarray(fit_baseline, dtype=float).copy()
        return population, population.copy()

    monkeypatch.setattr(pilot, "fit_pilot", guarded_fit)
    monkeypatch.setattr(pilot, "_evaluate", guarded_evaluate)
    monkeypatch.setattr(pilot, "predict_population", cheap_prediction)
    monkeypatch.setattr(pilot, "_integration_check", lambda *_: {
        "passed": True, "scope": "stubbed in orchestration test, not a numerical check"})
    argv = ["--archive", str(prepare_args.archive), "--output", str(output),
            "--fullcourse", "--updates", "1", "--threads", "1"]
    assert pilot.main(argv) == 0
    assert seen == ["fit", "evaluate"]
    plan = json.loads((output / "run_plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    history = json.loads((output / "history.json").read_text())
    assert len(plan["fit_targets"]) == 4 and len(plan["test_targets"]) == 2
    assert not set(plan["fit_targets"]) & set(plan["test_targets"])
    assert plan["config"]["updates"] == len(history) == 1
    assert plan["fit_replicates"] == [0, 1] and plan["test_replicates"] == [0, 1, 2]
    assert report["loss_by_time"]["4.0"] == {
        "updates": 0, "first_10_mean": None, "last_10_mean": None}
    assert report["status"] == "completed"
    assert report["held_out_responses_and_controls_used_for_fitting"] is False
    assert set(report["held_metrics_by_target"]) == {str(q) for q in plan["test_targets"]}
    assert set(report["metrics"]["held_target"]) == {
        "latent_neural_ode", "no_response", "direct_only_transfer"}
    assert all("target_effect_mae" in metrics for metrics in report["metrics"]["held_target"].values())
    json.dumps(report, allow_nan=False)
    checksums = json.loads((output / "SHA256SUMS.json").read_text())
    assert set(checksums) == {p.name for p in output.iterdir()
                              if p.is_file() and p.name != "SHA256SUMS.json"}
    for name, checksum in checksums.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    for name, checksum in plan["source_sha256"].items():
        assert hashlib.sha256(Path(pilot.__file__).with_name(name).read_bytes()).hexdigest() == checksum
    with pytest.raises(SystemExit) as error:
        pilot.main(argv)
    assert error.value.code == 2  # A completed run may not be overwritten.
    assert seen == ["fit", "evaluate"]
    assert checksums == json.loads((output / "SHA256SUMS.json").read_text())


def test_modular_cli_uses_registered_32_16_split_and_same_learner_family(tmp_path, monkeypatch):
    """One real update checks 64-gene wiring, not large-run training quality."""
    from grn_experiment import q1_pilot as pilot
    from grn_experiment import q1_pilot_data as loading

    baseline = np.full((3, 2, 64), 5, dtype=np.int64)
    baseline[2] = 9999
    observations = {}
    for q in range(48):
        for duration in (.25, 4.):
            control = np.full((3, 2, 64), 5, dtype=np.int64)
            treated = control.copy()
            treated[..., q] -= 1
            if q >= 32:
                treated[:] = 1000  # Held responses must never enter the update.
            observations[Condition(q, .7, duration)] = ReplicatedSnapshot(treated, control)
    provenance = {"schema": "scmultisim-q1-modular-v1", "initial_targets": list(range(48)),
                  "registered_split": {"train": list(range(32)), "held": list(range(32, 48)),
                                       "registration": "topology_before_rna"}}
    data = loading.OpenedInitialData(baseline, observations, provenance)
    monkeypatch.setattr(loading, "load_modular_q1", lambda *_: data)
    args = SimpleNamespace(dataset=tmp_path / "public", fullcourse=False, updates=None,
                           seed=20261023, split_seed=20261022)
    loaded, fitting, held, config = pilot._prepare_experiment(args)
    assert loaded is data and args.fullcourse is True
    assert config.updates == 4800 and config.hidden == 64 and config.step == .25
    assert config.target_weight == 0 and config.learning_rate == .001
    assert {key.target for key in fitting} == set(range(32))
    assert {key.target for key in held} == set(range(32, 48))
    output = tmp_path / "run"
    real_fit = pilot.fit_pilot
    seen = []

    def guarded_fit(fit_baseline, fit_observations, **kwargs):
        seen.append("fit")
        np.testing.assert_array_equal(fit_baseline, baseline[:2].reshape(-1, 64))
        assert {key.target for key in fit_observations} == set(range(32))
        assert all(value.fit_treated.max() <= 5 for value in fit_observations.values())
        return real_fit(fit_baseline, fit_observations, **kwargs)

    def cheap_prediction(model, fit_baseline, condition, **kwargs):
        assert (output / "model.pt").is_file()  # Checkpoint precedes held scoring.
        assert model.n_genes == 64
        population = np.asarray(fit_baseline, dtype=float).copy()
        return population, population.copy()

    monkeypatch.setattr(pilot, "fit_pilot", guarded_fit)
    monkeypatch.setattr(pilot, "predict_population", cheap_prediction)
    monkeypatch.setattr(pilot, "_integration_check", lambda *_: {
        "passed": True, "scope": "stubbed orchestration check, not numerical evidence"})
    assert pilot.main(["--dataset", str(args.dataset), "--output", str(output),
                       "--updates", "1", "--threads", "1"]) == 0
    assert seen == ["fit"]
    plan = json.loads((output / "run_plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert plan["mode"] == "modular_fullcourse_target_holdout"
    assert plan["fit_targets"] == list(range(32))
    assert plan["test_targets"] == list(range(32, 48))
    assert plan["split_seed"] is None
    assert report["held_out_responses_and_controls_used_for_fitting"] is False
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    assert checkpoint["n_genes"] == 64 and checkpoint["hidden"] == 64


def test_cli_requires_exactly_one_data_source(tmp_path):
    from grn_experiment.q1_pilot import main
    for argv in (["--output", str(tmp_path / "none")],
                 ["--archive", "old.tar.gz", "--dataset", "new", "--output", str(tmp_path / "both")]):
        with pytest.raises(SystemExit) as error:
            main(argv)
        assert error.value.code == 2
