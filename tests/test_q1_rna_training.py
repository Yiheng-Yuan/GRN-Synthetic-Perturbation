"""Regression checks for two-arm fitting; not evidence of predictive benefit."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from grn_experiment import q1_pilot as pilot
from grn_experiment.protocol import Condition, ReplicatedSnapshot
from grn_experiment.q1_pilot_data import OpenedInitialData


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs():
    rng = np.random.default_rng(3)
    baseline = rng.poisson(5, (3, 8, 4)).astype(float)
    observations = {}
    for time in (.25, 2.):
        for target in (0, 1):
            control = rng.poisson(5, (3, 8, 4)).astype(float)
            treated = control.copy()
            treated[..., target] = np.maximum(0, treated[..., target] - time)
            observations[Condition(target, .7, time)] = ReplicatedSnapshot(treated, control)
    return baseline, observations


def config(kind, **kwargs):
    return pilot.PilotConfig(updates=4, cells=4, projections=4, hidden=8,
        fit_times=(.25, 2.), condition_microbatch=1, model_kind=kind, **kwargs)


@pytest.mark.parametrize("kind", ["rna_current", "rna_history"])
def test_no_latent_noise_third_repeat_or_history_leakage(tmp_path, kind):
    baseline, observations = inputs()
    fit_baseline = baseline[:2].reshape(-1, 4)
    first, history = pilot.fit_pilot(fit_baseline, observations, config=config(kind))
    poisoned = {}
    for key, value in observations.items():
        a, b = value.treated.copy(), value.matched_control.copy()
        a[2], b[2] = 9999, 999
        poisoned[key] = ReplicatedSnapshot(a, b)
    second, other = pilot.fit_pilot(fit_baseline, poisoned, config=config(kind))
    assert history == other
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    assert not any("encoder" in key or "memory" in key for key in first.state_dict())
    a, b = pilot.predict_population(first, fit_baseline, Condition(3, .7, 2.), cells=8)
    assert a.shape == b.shape == (8, 4)
    assert np.isfinite(a).all() and np.isfinite(b).all() and (a >= 0).all()


@pytest.mark.parametrize("kind", ["rna_current", "rna_history"])
def test_interrupted_training_resumes_exactly_and_rejects_other_arm(tmp_path, kind):
    baseline, observations = inputs()
    baseline = baseline[:2].reshape(-1, 4)
    cfg = config(kind)
    expected, history = pilot.fit_pilot(baseline, observations, config=cfg)
    path = tmp_path / "training.pt"

    def interrupt(row):
        if row["update"] == 3:
            raise RuntimeError("interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        pilot.fit_pilot(baseline, observations, config=cfg, checkpoint_path=path,
                        checkpoint_every=2, on_update=interrupt)
    saved = path.read_bytes()
    wrong_kind = "rna_history" if kind == "rna_current" else "rna_current"
    with pytest.raises(ValueError, match="incompatible"):
        pilot.fit_pilot(baseline, observations, config=replace(cfg, model_kind=wrong_kind),
                        checkpoint_path=path, resume=True)
    assert path.read_bytes() == saved
    restored, resumed = pilot.fit_pilot(baseline, observations, config=cfg,
        checkpoint_path=path, checkpoint_every=2, resume=True)
    assert history == resumed
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_projection_directions_and_data_rng_are_same_despite_different_widths(tmp_path):
    baseline, observations = inputs()
    snapshots = []
    for kind, hidden in (("rna_current", 8), ("rna_history", 5)):
        path = tmp_path / f"{kind}.pt"
        pilot.fit_pilot(baseline[:2].reshape(-1, 4), observations,
            config=replace(config(kind), hidden=hidden), checkpoint_path=path)
        snapshots.append(torch.load(path, weights_only=True))
    torch.testing.assert_close(snapshots[0]["directions"], snapshots[1]["directions"], rtol=0, atol=0)
    assert snapshots[0]["numpy_rng"] == snapshots[1]["numpy_rng"]


@pytest.mark.parametrize("kind", ["rna_current", "rna_history"])
def test_cli_saves_correct_model_identity_sources_predictions_and_checks(tmp_path, monkeypatch, kind):
    baseline, fitting = inputs()
    held = {Condition(3, .7, 2.): next(iter(fitting.values()))}
    data = OpenedInitialData(baseline, {**fitting, **held}, {"phase": "development"})
    monkeypatch.setattr(pilot, "_prepare_experiment", lambda args:
                        (data, fitting, held, config(kind)))
    output = tmp_path / kind
    assert pilot.main(["--dataset", "unused", "--output", str(output),
                       "--model-kind", kind, "--fullcourse", "--threads", "1",
                       "--checkpoint-every", "2"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["model_kind"] == kind
    assert "q1_rna.py" in report["source_snapshot"]["files"]
    assert report["state_contract"]["latent_noise"] is False
    assert report["integration_check"]["passed"] is True
    weights = torch.load(output / "model.pt", weights_only=True)
    reloaded = pilot.build_model(weights["n_genes"], pilot.PilotConfig(**{
        **weights["config"], "fit_times": tuple(weights["config"]["fit_times"])}))
    reloaded.load_state_dict(weights["state_dict"], strict=True)
    with np.load(output / "predictions.npz", allow_pickle=False) as predictions:
        assert any(key.startswith(kind + "_") for key in predictions)
        assert not any("latent_neural_ode" in key for key in predictions)
    # The comparison launcher must accept real trainer output, not only a
    # hand-written approximation of its schema.
    from test_q1_rna_delivery import runner
    entry = runner()
    registration = {"configs": {kind: report["config"]},
                    "input_provenance": report["input_provenance"],
                    "fit_targets": report["fit_targets"], "held_targets": report["test_targets"],
                    "sources": {"src/grn_experiment/" + name: digest
                                for name, digest in report["source_sha256"].items()}}
    assert entry.verify_completed(output, registration, kind)["status"] == "completed"


def test_image_widths_keep_parameter_counts_close():
    current = pilot.build_model(64, pilot.PilotConfig(model_kind="rna_current", hidden=64))
    delayed = pilot.build_model(64, pilot.PilotConfig(model_kind="rna_history", hidden=51))
    counts = [sum(p.numel() for p in model.parameters()) for model in (current, delayed)]
    assert abs(counts[0] - counts[1]) / counts[0] < .02


@pytest.mark.parametrize("change", [{"model_kind": "unknown"}, {"history_lag": 0},
                                   {"history_lag": float("nan")}, {"hidden": 0}])
def test_bad_config_rejected(change):
    with pytest.raises(ValueError):
        pilot.PilotConfig(**change)
