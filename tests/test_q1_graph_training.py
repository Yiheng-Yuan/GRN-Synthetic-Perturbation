"""Tiny graph-training and data-isolation contracts; no research training."""

from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest
import torch

from grn_experiment import q1_graph_train as trainer
from grn_experiment.protocol import Condition, ReplicatedSnapshot
from grn_experiment.q1_pilot_data import OpenedInitialData, load_modular_q1
from test_q1_pilot_data import _modular_fixture


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(*, targets=2, genes=4):
    rng = np.random.default_rng(42)
    baseline = rng.poisson(5, (3, 8, genes)).astype(np.float32)
    observations = {}
    for time in (.25, .5):
        for target in range(targets):
            control = rng.poisson(5, (3, 8, genes)).astype(float)
            treated = control.copy()
            treated[..., target] = np.maximum(0, treated[..., target] - time)
            observations[Condition(target, .7, time)] = ReplicatedSnapshot(treated, control)
    return baseline, observations


def config(**kwargs):
    return trainer.GraphConfig(updates=4, cells=4, projections=3, hidden=4,
        conditions_per_update=2, condition_microbatch=1, fit_times=(.25, .5), **kwargs)


def poisoned_check(observations):
    result = {}
    for key, value in observations.items():
        a, b = value.treated.copy(), value.matched_control.copy()
        a[2], b[2] = 9999, 999
        result[key] = ReplicatedSnapshot(a, b)
    return result


def test_population_loss_is_absolute_expression_and_distribution_without_delta():
    # A common expression shift leaves treated-minus-control response exactly
    # unchanged, but must be penalized by the absolute-expression objective.
    obs_c = torch.tensor([[[1., 2.], [3., 4.]]])
    obs_t = obs_c + 2.
    scale = torch.ones(2)
    directions = torch.eye(2)
    losses = trainer.population_losses(obs_t + 1., obs_c + 1., obs_t, obs_c,
        scale, scale, directions)
    assert set(losses) == {"expression", "expression_treated", "expression_control", "distribution"}
    assert losses["expression"].item() == 1.
    assert losses["distribution"].item() > 0
    exact = trainer.population_losses(obs_t, obs_c, obs_t, obs_c,
        scale, scale, directions)
    assert all(value.item() == 0 for value in exact.values())
    # Independent population row order is irrelevant; no fake cell pairing.
    permuted = trainer.population_losses(obs_t + 1., obs_c + 1., obs_t.flip(-2),
        obs_c.flip(-2), scale, scale, directions)
    for key in losses:
        torch.testing.assert_close(losses[key], permuted[key], atol=0, rtol=0)


def test_graph_training_logs_the_actual_fixed_weights_and_useful_gradients():
    baseline, observations = inputs()
    cfg = config()
    model, history = trainer.fit_graph(baseline[:2].reshape(-1, 4), observations, config=cfg)
    assert len(history) == cfg.updates
    assert [row["time"] for row in history] == [.25, .5, .25, .5]
    for row in history:
        assert row["weighted_distribution"] == pytest.approx(cfg.lambda_distribution * row["distribution"])
        assert row["weighted_sparse"] == pytest.approx(cfg.lambda_sparse * row["expected_edges"])
        assert row["loss"] == pytest.approx(row["expression"] + row["weighted_distribution"] + row["weighted_sparse"])
        assert row["gradient_norms"]["raw_H"] > 0
        assert row["gradient_norms"]["gate_logits"] > 0
        assert row["gradient_norms"]["raw_rho"] > 0
        assert row["gradient_norms"]["receiver_weight"] > 0
    assert not any("lambda" in name for name, _ in model.named_parameters())
    with pytest.raises(FrozenInstanceError):
        cfg.lambda_sparse = 0
    assert not model.training
    matrices = model.graph_matrices()
    assert set(matrices["W"].unique().tolist()) <= {0., 1.}
    torch.testing.assert_close(matrices["Z"], matrices["W"] * matrices["H"], atol=0, rtol=0)


def test_poisoning_third_repeats_cannot_change_fitting_or_checkpoint_identity():
    baseline, observations = inputs()
    fit_baseline = baseline[:2].reshape(-1, 4)
    poisoned = poisoned_check(observations)
    assert trainer._input_digest(fit_baseline, observations) == trainer._input_digest(fit_baseline, poisoned)
    model, history = trainer.fit_graph(fit_baseline, observations, config=config())
    other, other_history = trainer.fit_graph(fit_baseline, poisoned, config=config())
    assert history == other_history
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[name], atol=0, rtol=0)


def test_interrupted_graph_training_resumes_exactly_and_rejects_changed_weight(tmp_path):
    baseline, observations = inputs()
    baseline = baseline[:2].reshape(-1, 4)
    cfg = config()
    expected, history = trainer.fit_graph(baseline, observations, config=cfg)
    path = tmp_path / "training.pt"

    def interrupt(row):
        if row["update"] == 3:
            raise RuntimeError("test interruption")

    with pytest.raises(RuntimeError, match="test interruption"):
        trainer.fit_graph(baseline, observations, config=cfg, checkpoint_path=path,
            checkpoint_every=2, on_update=interrupt)
    payload = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        trainer.fit_graph(baseline, observations, config=replace(cfg, lambda_sparse=1e-3),
            checkpoint_path=path, resume=True)
    assert path.read_bytes() == payload
    restored, resumed = trainer.fit_graph(baseline, observations, config=cfg,
        checkpoint_path=path, checkpoint_every=2, resume=True)
    assert history == resumed
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)


def test_internal_validation_is_grouped_by_target_not_cells_or_times():
    _, observations = inputs(targets=10, genes=12)
    fitting, validation = trainer.internal_split(observations, validation_count=3)
    fit_ids, val_ids = ({k.target for k in group} for group in (fitting, validation))
    assert len(fit_ids) == 7 and len(val_ids) == 3 and fit_ids.isdisjoint(val_ids)
    assert set(fitting) | set(validation) == set(observations)
    for target in fit_ids | val_ids:
        group = fitting if target in fit_ids else validation
        assert {key.time for key in group if key.target == target} == {.25, .5}
    poisoned = poisoned_check(dict(reversed(list(observations.items()))))
    again = trainer.internal_split(poisoned, validation_count=3)
    assert [set(group) for group in again] == [set(fitting), set(validation)]


def test_numerical_diagnostic_uses_same_hard_graph_and_flags_large_step_error(monkeypatch):
    baseline, observations = inputs()
    baseline = baseline[:2].reshape(-1, 4)
    model, _ = trainer.fit_graph(baseline, observations, config=config())
    report = trainer.integration_check(model, baseline, observations, config())
    assert report["passed"] and report["time"] == .5
    assert report["fine_step"] == report["coarse_step"] / 2
    calls = []

    def inaccurate(model, baseline, key, *, cells, seed, step):
        calls.append((key, cells, seed, step))
        values = torch.full((cells, 4), 10. if step == .25 else 1.)
        return values, values

    monkeypatch.setattr(trainer, "predict", inaccurate)
    assert trainer.integration_check(model, baseline, observations, config())["passed"] is False
    assert calls[0][:3] == calls[1][:3]


def test_train_scope_can_load_with_all_held_responses_absent(tmp_path):
    directory = _modular_fixture(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    held = set(manifest["q1_split"]["held"])
    for entry in manifest["conditions"].values():
        if entry["condition"]["target"] in held:
            (directory / entry["observation"]).unlink()
    train = load_modular_q1(directory, scope="train")
    assert {key.target for key in train.observations} == set(range(32))
    assert train.provenance["verified_observation_count"] == 193
    assert train.provenance["decoded_private_files"] == 0
    with pytest.raises((ValueError, FileNotFoundError)):
        load_modular_q1(directory, scope="all")


def _mock_public_data(monkeypatch):
    baseline, observations = inputs(targets=10, genes=12)
    scopes = []
    data = OpenedInitialData(baseline, observations, {"phase": "development",
        "registered_split": {"train": list(range(10)), "held": [10, 11]}})

    def load(path, *, scope):
        scopes.append(scope)
        assert scope == "train", "selection must not decode held responses"
        return data

    monkeypatch.setattr(trainer, "load_modular_q1", load)
    return data, scopes


def test_cli_probe_runs_one_longest_time_update_and_never_claims_full_training(tmp_path, monkeypatch):
    _, scopes = _mock_public_data(monkeypatch)
    output = tmp_path / "probe"
    assert trainer.main(["--dataset", "unused", "--output", str(output), "--device", "cpu",
        "--probe", "--updates", "4", "--cells", "4", "--hidden", "4", "--threads", "1"]) == 0
    report = json.loads((output / "probe.json").read_text())
    plan = json.loads((output / "run_plan.json").read_text())
    assert scopes == ["train"]
    assert report["status"] == "passed" and report["finite_loss"] is True
    assert report["config"]["updates"] == 1 and report["config"]["fit_times"] == [.5]
    assert plan["held_responses_decoded_for_selection"] is False
    assert not (output / "report.json").exists()
    assert not (output / "model.pt").exists()
    assert len((output / "progress.jsonl").read_text().splitlines()) == 1


def test_cli_locks_binary_graph_and_model_before_internal_validation(tmp_path, monkeypatch):
    _, scopes = _mock_public_data(monkeypatch)
    output = tmp_path / "fit"
    original_validate = trainer.validation_metrics

    def validate(model, baseline, observations, cfg):
        lock = json.loads((output / "prediction_lock.json").read_text())
        assert all(trainer.sha256(output / name) == digest for name, digest in lock.items())
        assert not model.training
        return original_validate(model, baseline, observations, cfg)

    monkeypatch.setattr(trainer, "validation_metrics", validate)
    assert trainer.main(["--dataset", "unused", "--output", str(output), "--device", "cpu",
        "--updates", "2", "--cells", "4", "--hidden", "4", "--threads", "1"]) == 0
    assert scopes == ["train"]
    report = json.loads((output / "report.json").read_text())
    lock = json.loads((output / "prediction_lock.json").read_text())
    assert report["status"] == "completed" and report["metrics"] is None
    assert report["validation_expression"] >= 0
    assert set(report["fit_targets"]).isdisjoint(report["validation_targets"])
    assert not set(report["held_targets"]) & (set(report["fit_targets"]) | set(report["validation_targets"]))
    with np.load(output / "learned_graph.npz", allow_pickle=False) as graph:
        assert set(np.unique(graph["W"])) <= {0, 1}
        np.testing.assert_array_equal(graph["Z"], graph["W"] * graph["H"])
        assert graph["orientation"].item() == "row_receiver_col_source"
    assert all(trainer.sha256(output / name) == digest for name, digest in lock.items())
    assert (output / "source_snapshot/q1_graph.py").is_file()
    checksums = json.loads((output / "SHA256SUMS.json").read_text())
    assert checksums["learned_graph.npz"] == trainer.sha256(output / "learned_graph.npz")


@pytest.mark.parametrize("change", [
    {"lambda_sparse": -.1}, {"lambda_distribution": float("nan")},
    {"lambda_sparse": True}, {"step": 0}, {"fit_times": (.5, .25)},
])
def test_invalid_graph_config_rejected(change):
    with pytest.raises(ValueError):
        replace(config(), **change)
