"""Entry/resume acceptance checks; mocked GPU tests are not CUDA validation."""

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


PROJECT = Path(__file__).parents[1]


def runner():
    directory = PROJECT / "tools/scmultisim"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("rna_entry", directory / "q1_rna_run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def registered(module):
    return {"configs": {kind: json.loads(json.dumps(asdict(module.arm_config(
        kind, hidden, (.25, 1., 2., 4., 8., 64.))))) for kind, hidden in module.ARMS},
        "manifest_sha256": module.MANIFEST_SHA256, "sources": {},
        "input_provenance": {"public": True}, "fit_targets": [0], "held_targets": [1],
        "scope": "test fixture", "parameter_counts": {"rna_current": 1, "rna_history": 1}}


def test_shell_syntax_and_bad_arguments():
    entry = PROJECT / "tools/scmultisim/run_q1_rna.sh"
    subprocess.run(["bash", "-n", str(entry)], check=True)
    result = subprocess.run(["bash", str(entry), "unknown"], capture_output=True, text=True)
    assert result.returncode == 64


def test_cpu_host_fails_explicitly_and_preserves_log(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CPU refusal test requires no CUDA GPU")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    output = tmp_path / "persistent"
    output.mkdir()
    entry = PROJECT / "tools/scmultisim/run_q1_rna.sh"
    env = {**os.environ, "Q1_DATASET": str(data), "Q1_OUTPUT_ROOT": str(output),
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", str(entry)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "refusing an accidental CPU" in result.stdout
    folder = output / "q1-rna-comparison-seed20261022"
    assert "Task exit status: 1" in (folder / "run.log").read_text()
    assert len(list(folder.glob("failure-*.json"))) == 1
    before = (folder / "run.log").read_bytes()
    second = subprocess.run(["bash", str(entry)], env=env, capture_output=True, text=True)
    assert second.returncode != 0
    assert (folder / "run.log").read_bytes() == before


def test_runner_sequentially_probes_and_trains_exact_registered_arms(tmp_path, monkeypatch):
    module = runner()
    plan = registered(module)
    calls = []
    monkeypatch.setattr(module, "registration", lambda _: plan)
    monkeypatch.setattr(module, "probe_arm", lambda *args: calls.append(("probe", args[-2], args[-1])))
    monkeypatch.setattr(module, "train_arm", lambda *args: calls.append(("train", args[-3], args[-2])))
    monkeypatch.setattr(module, "verify_completed", lambda *_: {"metrics": {}, "seconds": 0.})
    module.run(SimpleNamespace(mode="train", dataset=tmp_path, output=tmp_path))
    assert calls == [("probe", "rna_current", 64), ("train", "rna_current", 64),
                     ("probe", "rna_history", 51), ("train", "rna_history", 51)]
    assert json.loads((tmp_path / "comparison.json").read_text())["status"] == "completed"


def test_resume_changed_source_or_data_refused_before_probe_or_training(tmp_path, monkeypatch):
    module = runner()
    plan = registered(module)
    (tmp_path / "comparison_plan.json").write_text(json.dumps({**plan, "sources": {"changed": "x"}}))
    monkeypatch.setattr(module, "registration", lambda _: plan)
    monkeypatch.setattr(module, "probe_arm", lambda *_: pytest.fail("probe after incompatible resume"))
    monkeypatch.setattr(module, "train_arm", lambda *_: pytest.fail("train after incompatible resume"))
    with pytest.raises(ValueError, match="registration changed"):
        module.run(SimpleNamespace(mode="resume", dataset=tmp_path, output=tmp_path))


def test_failed_probe_cannot_trigger_training(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module, "registration", lambda _: registered(module))
    monkeypatch.setattr(module, "run_probe", lambda *_args, **_kwargs: {"status": "failed"})
    monkeypatch.setattr(module, "train_arm", lambda *_: pytest.fail("training after failed probe"))
    with pytest.raises(RuntimeError, match="GPU acceptance failed"):
        module.run(SimpleNamespace(mode="train", dataset=tmp_path, output=tmp_path))


def test_completed_arm_checksum_corruption_is_not_skipped(tmp_path):
    module = runner()
    (tmp_path / "report.json").write_text('{"status": "completed"}')
    (tmp_path / "run_plan.json").write_text("{}")
    names = ("model.pt", "training_checkpoint.pt", "report.json", "run_plan.json",
             "history.json", "scores.json", "predictions.npz", "progress.jsonl")
    hashes = {}
    for name in names:
        path = tmp_path / name
        if not path.exists():
            path.write_text("fixture")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS.json").write_text(json.dumps(hashes))
    (tmp_path / "model.pt").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.verify_completed(tmp_path, registered(module), "rna_current")


def test_incomplete_arm_without_checkpoint_is_never_overwritten(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("overwrite"))
    with pytest.raises(ValueError, match="no resumable checkpoint"):
        module.train_arm(tmp_path, tmp_path, "rna_current", 64, True)


def test_resume_passes_original_budget_and_architecture_to_trainer(tmp_path, monkeypatch):
    module = runner()
    (tmp_path / "training_checkpoint.pt").write_text("fixture")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    module.train_arm(tmp_path, tmp_path, "rna_history", 51, True)
    argv, kwargs = calls[0]
    assert argv[argv.index("--model-kind") + 1] == "rna_history"
    assert argv[argv.index("--hidden") + 1] == "51"
    assert argv[argv.index("--updates") + 1] == "4800"
    assert argv[-1] == "--resume" and kwargs["check"] is True
