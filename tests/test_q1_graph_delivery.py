"""Runner contract checks with mocked subprocesses, not GPU acceptance."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT = Path(__file__).parents[1]


def runner():
    spec = importlib.util.spec_from_file_location(
        "graph_entry", PROJECT / "tools/scmultisim/q1_graph_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registration(module):
    return {"configs": {name: module.arm_config(weight, 4) for name, weight in module.ARMS},
            "manifest_sha256": module.MANIFEST_SHA256, "sources": {}, "runtime": {}, "n_genes": 3,
            "scope": "test fixture", "selection": {
                "metric": "validation_expression", "absolute_tie_tolerance": 1e-6,
                "tie_breaks": ["fewest_binary_edges", "smaller_lambda_sparse"]}}


def complete_fixture(module, directory, registered, name, *, score=.2, edges=1,
                     refit=False):
    directory.mkdir(exist_ok=True)
    plan = {"config": registered["configs"][name], "refit": refit,
            "source_sha256": {Path(path).name: digest for path, digest in registered["sources"].items()
                              if path.startswith("src/grn_experiment/")},
            "runtime": registered.get("runtime", {}),
            "provenance": {"manifest_sha256": registered["manifest_sha256"]}}
    (directory / "run_plan.json").write_text(json.dumps(plan))
    (directory / "report.json").write_text(json.dumps({
        **plan, "status": "completed", "validation_expression": None if refit else score,
        "held_expression": -1000.0}))  # Must never select by held-target score.
    w = np.zeros((3, 3))
    w[1, 0] = 1
    if edges == 2:
        w[2, 1] = 1
    h = np.ones_like(w)
    np.savez(directory / "learned_graph.npz", W=w, H=h, Z=w*h)
    (directory / "model.pt").write_bytes(b"fixture")
    names = ("run_plan.json", "report.json", "learned_graph.npz", "model.pt")
    (directory / "SHA256SUMS.json").write_text(json.dumps({
        path: module.checksum(directory / path) for path in names}))


def test_shell_syntax_and_bad_arguments():
    entry = PROJECT / "tools/scmultisim/run_q1_graph.sh"
    subprocess.run(["bash", "-n", str(entry)], check=True)
    result = subprocess.run(["bash", str(entry), "invalid"], capture_output=True)
    assert result.returncode == 64
    result = subprocess.run(["bash", str(entry), "--help"], capture_output=True)
    assert result.returncode == 0 and b"Q1_OUTPUT_ROOT" in result.stdout


def test_commands_keep_each_sparse_weight_and_probe_separate(tmp_path):
    module = runner()
    for name, weight in module.ARMS:
        command = module.trainer_command(tmp_path, tmp_path / name,
                                         module.arm_config(weight, 4800), resume=True)
        assert command[command.index("--lambda-sparse") + 1] == str(weight)
        assert command[command.index("--updates") + 1] == "4800"
        assert command[command.index("--device") + 1] == "cuda"
        assert "--resume" in command and "--probe" not in command


def test_changed_resume_registration_cannot_run_subprocess(tmp_path, monkeypatch):
    module = runner()
    registered = registration(module)
    (tmp_path / "comparison_plan.json").write_text(json.dumps({**registered, "sources": {"changed": "x"}}))
    monkeypatch.setattr(module, "registration", lambda *_: registered)
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: pytest.fail("subprocess after incompatible resume"))
    with pytest.raises(ValueError, match="registration changed"):
        module.run(SimpleNamespace(mode="resume", dataset=tmp_path, output=tmp_path, updates=4))


def test_training_is_sequential_and_uses_validation_only(tmp_path, monkeypatch):
    module = runner()
    registered = registration(module)
    calls = []
    monkeypatch.setattr(module, "registration", lambda *_: registered)
    monkeypatch.setattr(module, "probe", lambda *_: calls.append("probe"))

    def fake_training(command, **kwargs):
        name = Path(command[command.index("--output") + 1]).name
        calls.append(name)
        if name == "final_refit":
            assert "--refit" in command
            assert command[command.index("--lambda-sparse") + 1] == "1e-05"
            complete_fixture(module, tmp_path / name, registered, "sparse-1e-5", refit=True)
        else:
            complete_fixture(module, tmp_path / name, registered, name,
                             score={"sparse-0": .3, "sparse-1e-5": .1,
                                    "sparse-1e-4": .2, "sparse-1e-3": .4}[name])
        assert kwargs["check"] is True

    monkeypatch.setattr(module.subprocess, "run", fake_training)
    module.run(SimpleNamespace(mode="train", dataset=tmp_path, output=tmp_path, updates=4))
    assert calls == ["probe", *(name for name, _ in module.ARMS), "final_refit"]
    selected = json.loads((tmp_path / "selection.json").read_text())
    assert selected["selected_arm"] == "sparse-1e-5"
    assert all("held_expression" not in row for row in selected["candidates"].values())


def test_selection_tie_uses_edges_then_lambda():
    module = runner()
    reports = {"sparse-0": {"validation_expression": 1., "binary_edges": 2, "lambda_sparse": 0.},
               "sparse-1e-5": {"validation_expression": 1.+5e-7, "binary_edges": 1, "lambda_sparse": 1e-5},
               "sparse-1e-4": {"validation_expression": 1., "binary_edges": 1, "lambda_sparse": 1e-4}}
    assert module.select_arm(reports, registration(module))["selected_arm"] == "sparse-1e-5"


def test_probe_failure_prevents_any_full_training(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module, "registration", lambda *_: registration(module))
    calls = []

    def fail_probe(command, **kwargs):
        calls.append(command)
        path = Path(command[command.index("--output") + 1])
        path.mkdir()
        (path / "probe.json").write_text('{"status":"failed"}')

    monkeypatch.setattr(module.subprocess, "run", fail_probe)
    with pytest.raises(RuntimeError, match="probe failed"):
        module.run(SimpleNamespace(mode="train", dataset=tmp_path, output=tmp_path, updates=4))
    assert len(calls) == 1 and "--probe" in calls[0]


def test_corrupt_completed_arm_is_not_silently_skipped(tmp_path):
    module = runner()
    registered = registration(module)
    complete_fixture(module, tmp_path, registered, "sparse-0")
    (tmp_path / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.verify_completed(tmp_path, registered, "sparse-0")


def test_resume_completed_runs_verifies_without_training(tmp_path, monkeypatch):
    module = runner()
    registered = registration(module)
    (tmp_path / "comparison_plan.json").write_text(json.dumps(registered))
    for name, _ in module.ARMS:
        complete_fixture(module, tmp_path / name, registered, name)
    complete_fixture(module, tmp_path / "final_refit", registered, "sparse-0", refit=True)
    monkeypatch.setattr(module, "registration", lambda *_: registered)
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: pytest.fail("repeat training"))
    module.run(SimpleNamespace(mode="resume", dataset=tmp_path, output=tmp_path, updates=4))
    assert (tmp_path / "selection.json").is_file()


def test_refit_verification_rejects_a_validation_run(tmp_path):
    module = runner()
    registered = registration(module)
    complete_fixture(module, tmp_path, registered, "sparse-0")
    with pytest.raises(ValueError, match="incomplete run"):
        module.verify_completed(tmp_path, registered, "sparse-0", refit=True)


def test_cpu_failure_preserves_log_and_new_run_cannot_overwrite(tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip("CPU refusal requires a CPU host")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    output = tmp_path / "results"
    entry = PROJECT / "tools/scmultisim/run_q1_graph.sh"
    env = {**os.environ, "Q1_DATASET": str(data), "Q1_OUTPUT_ROOT": str(output),
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    first = subprocess.run(["bash", str(entry), "probe"], env=env,
                           capture_output=True, text=True)
    assert first.returncode != 0 and "refusing an accidental CPU" in first.stdout
    directory = output / "q1-graph-probe-seed20261022"
    before = (directory / "run.log").read_bytes()
    assert len(list(directory.glob("failure-*.json"))) == 1
    second = subprocess.run(["bash", str(entry), "probe"], env=env,
                            capture_output=True, text=True)
    assert second.returncode != 0
    assert before == (directory / "run.log").read_bytes()


@pytest.mark.parametrize("refit", (False, True))
def test_real_trainer_artifacts_match_runner_contract(tmp_path, monkeypatch, refit):
    """A one-update CPU fixture checks interfaces, not training performance."""
    from grn_experiment import q1_graph_train as trainer
    from grn_experiment.protocol import Condition, ReplicatedSnapshot
    from grn_experiment.q1_pilot_data import OpenedInitialData

    module = runner()
    counts = np.full((3, 2, 12), 4, dtype=np.int64)
    observations = {}
    for target in range(12):
        treated = counts.copy()
        treated[..., target] = 3
        for duration in (.25, .5):
            observations[Condition(target, .7, duration)] = ReplicatedSnapshot(treated, counts.copy())
    provenance = {"manifest_sha256": module.MANIFEST_SHA256,
                  "registered_split": {"train": list(range(10)), "held": [10, 11]}}
    scopes = []

    def loader(_path, *, scope):
        scopes.append(scope)
        selected = observations if scope == "all" else {
            key: value for key, value in observations.items() if key.target < 10}
        return OpenedInitialData(counts.copy(), selected, provenance)

    monkeypatch.setattr(trainer, "load_modular_q1", loader)
    output = tmp_path / "actual"
    command = ["--dataset", str(tmp_path / "fixture-data"), "--output", str(output),
               "--updates", "1", "--cells", "2", "--hidden", "2", "--threads", "1",
               "--conditions-per-update", "1", "--device", "cpu"]
    if refit:
        command.append("--refit")
    assert trainer.main(command) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["refit"] is refit
    assert scopes == (["train", "all"] if refit else ["train"])
    plan = registration(module)
    plan["configs"]["sparse-1e-4"] = report["config"]
    plan["n_genes"] = 12
    plan["sources"] = {"src/grn_experiment/" + name: digest
                       for name, digest in trainer.source_hashes().items()}
    plan["runtime"] = report["runtime"]
    verified = module.verify_completed(output, plan, "sparse-1e-4", refit=refit)
    assert verified["binary_edges"] == report["binary_edges"]
    assert verified["validation_expression"] == report["validation_expression"]
    assert set(trainer.source_hashes()) == {
        Path(path).name for path in module.SOURCE_PATHS if path.startswith("src/")}


@pytest.mark.parametrize("metadata", ("report.json", "run_plan.json"))
@pytest.mark.parametrize("field", ("source_sha256", "runtime"))
def test_rehashed_foreign_arm_metadata_is_rejected(tmp_path, metadata, field):
    module = runner()
    registered = registration(module)
    registered["sources"] = {"src/grn_experiment/q1_graph.py": "a"*64}
    registered["runtime"] = {"python": "3.12.15", "numpy": "2.5.3", "torch": "2.14.0",
                             "machine": "x86_64", "cuda": "12.6", "gpu": "test_gpu",
                             "capability": [8, 9]}
    complete_fixture(module, tmp_path, registered, "sparse-0")
    value = json.loads((tmp_path / metadata).read_text())
    if field == "source_sha256":
        value[field]["q1_graph.py"] = "b"*64
    else:
        value[field]["torch"] = "foreign_version"
    (tmp_path / metadata).write_text(json.dumps(value))
    hashes = json.loads((tmp_path / "SHA256SUMS.json").read_text())
    hashes[metadata] = module.checksum(tmp_path / metadata)
    (tmp_path / "SHA256SUMS.json").write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="differs from registration"):
        module.verify_completed(tmp_path, registered, "sparse-0")
