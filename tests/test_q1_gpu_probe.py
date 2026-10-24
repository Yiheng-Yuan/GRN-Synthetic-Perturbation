"""CPU-only contracts for the GPU probe; no actual long-time GPU fit here."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from grn_experiment.protocol import COMMON_BASELINE, Condition


PROJECT = Path(__file__).parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "q1_gpu_probe", PROJECT / "tools/scmultisim/q1_gpu_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostic(target, dose):
    return {"passed": True, "burnin": 48., "report": {
        "target": target, "dose": dose, "tau": 1.,
        "schema_version": "scmultisim-stationarity-v1",
        "status": "diagnostic_passed", "parameter_sha256": "a" * 64}}


def _dataset(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "scmultisim-q1-modular-v1", "phase": "development",
        "formal_calibration": False, "status": "complete", "n_genes": 64,
        "cells_per_replicate": 2, "target_ids": list(range(48)),
        "q1_split": {"train": list(range(32)), "held": list(range(32, 48)),
                     "registration": "topology_before_rna"},
        "sampling_times": [.25, 1., 2., 4., 8.], "dose": .7,
        "common_longterm_time": 64., "baseline_stationarity": _diagnostic(None, 0.),
        "conditions": {},
    }
    for condition in (COMMON_BASELINE, *(Condition(q, .7, 64.) for q in range(4))):
        record = {"kind": "timecourse", "target": condition.target,
                  "dose": condition.dose, "time": condition.time}
        identifier = hashlib.sha256(json.dumps(record, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()[:24]
        relative = f"observations/q1/{identifier}.npz"
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        treated = np.full((3, 2, 64), 5, dtype=np.int64)
        treated[2] = 777  # Check repeat 2 is excluded from the fit inputs.
        arrays = {"treated": treated}
        if condition.target is not None:
            arrays["matched_control"] = treated.copy()
            arrays["treated"][:2, :, condition.target] -= 1
        np.savez_compressed(path, **arrays)
        entry = {"condition": record, "status": "complete", "observation": relative,
                 "files": {relative: hashlib.sha256(path.read_bytes()).hexdigest()}}
        if condition.target is not None:
            entry.update(actual_sample_time=64., diagnostic=_diagnostic(condition.target, .7))
        manifest["conditions"][identifier] = entry
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _fake_cuda(module, monkeypatch):
    cuda = module.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "get_device_name", lambda *_: "Fixture RTX 4070")
    monkeypatch.setattr(cuda, "get_device_capability", lambda *_: (8, 9))
    monkeypatch.setattr(cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(cuda, "mem_get_info", lambda *_: (6_000_000_000, 8_000_000_000))
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda *_: 1_500_000_000)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda *_: 2_000_000_000)
    monkeypatch.setattr(module.torch, "set_num_threads", lambda *_: None)


def test_probe_loads_only_baseline_and_four_registered_training_long_endpoints(tmp_path, monkeypatch):
    module = _module()
    manifest = _dataset(tmp_path)
    allowed = {"manifest.json", *(entry["observation"] for entry in manifest["conditions"].values())}
    opened, original = [], module._directory_payload

    def guard(directory, relative, limit):
        assert relative in allowed
        opened.append(relative)
        return original(directory, relative, limit)

    monkeypatch.setattr(module, "_directory_payload", guard)
    baseline, observations, provenance = module.load_probe_data(tmp_path)
    assert len(opened) == 6 and set(opened) == allowed
    assert baseline.shape == (4, 64) and np.all(baseline == 5)
    assert {key.target for key in observations} == set(range(4))
    assert {key.time for key in observations} == {64.}
    assert all(np.max(value.fit_treated) == 5 for value in observations.values())
    assert all(np.max(value.fit_control) == 5 for value in observations.values())
    assert provenance["verified_observation_count"] == 5
    assert provenance["manifest_sha256"] == hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    assert provenance["held_targets_read"] is False
    assert provenance["third_replicate_used_for_fitting"] is False


@pytest.mark.parametrize("change", ["checksum", "actual_time", "diagnostic", "missing"])
def test_probe_rejects_inconsistent_selected_data(tmp_path, change):
    module = _module()
    manifest = _dataset(tmp_path)
    entry = next(entry for entry in manifest["conditions"].values()
                 if entry["condition"]["target"] is not None)
    if change == "checksum":
        (tmp_path / entry["observation"]).write_bytes(b"corrupt payload")
    elif change == "actual_time":
        entry["actual_sample_time"] = 63.
    elif change == "diagnostic":
        entry["diagnostic"]["report"]["parameter_sha256"] = "b" * 64
    else:
        (tmp_path / entry["observation"]).unlink()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.load_probe_data(tmp_path)


def test_probe_measures_full_effective_update_without_changing_model_settings(tmp_path, monkeypatch):
    module = _module()
    _dataset(tmp_path / "data")
    _fake_cuda(module, monkeypatch)
    calls = []

    def fake_fit(baseline, observations, *, config, device):
        calls.append(config)
        assert baseline.shape == (4, 64) and baseline.max() == 5
        assert len(observations) == 4 and device == "cuda:0"
        assert config.updates == 1 and config.conditions_per_update == 4
        assert config.condition_microbatch == 1 and config.cells == 64
        assert config.hidden == 64 and config.projections == 24 and config.step == .25
        assert config.learning_rate == .001 and config.effect_weight == 1. and config.target_weight == 0.
        assert config.seed == 20261022 and config.fit_times == (64.,)
        return None, [{"update": 1, "time": 64., "loss": .5}]

    monkeypatch.setattr(module, "fit_pilot", fake_fit)
    path = tmp_path / "probe.json"
    result = module.run_probe(tmp_path / "data", path)
    assert len(calls) == 1 and json.loads(path.read_text()) == result
    assert result["status"] == "passed" and result["finite_loss"] is True
    assert result["target_ids"] == [0, 1, 2, 3] and result["actual_time"] == 64.
    assert result["device_name"] == "Fixture RTX 4070" and result["cuda_capability"] == [8, 9]
    assert result["peak_allocated_bytes"] == 1_500_000_000
    assert result["peak_reserved_bytes"] == 2_000_000_000
    assert result["free_bytes"] == 6_000_000_000 and result["total_bytes"] == 8_000_000_000
    assert result["elapsed_seconds"] >= 0 and len(result["source_sha256"]) == 3
    assert all(result[key] for key in ("python_version", "numpy_version", "torch_version"))


def test_probe_requires_gpu_and_never_overwrites_a_report(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, "fit_pilot", lambda *_args, **_kwargs: pytest.fail("CPU fallback"))
    path = tmp_path / "probe.json"
    result = module.run_probe(tmp_path / "missing-data", path)
    assert result["status"] == "failed" and "refusing an accidental CPU" in result["error"]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.run_probe(tmp_path / "missing-data", path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", ["oom", "nonfinite"])
def test_probe_preserves_failure_report_instead_of_claiming_pass(tmp_path, monkeypatch, failure):
    module = _module()
    _dataset(tmp_path / "data")
    _fake_cuda(module, monkeypatch)

    def failed_fit(*_args, **_kwargs):
        if failure == "oom":
            raise module.torch.cuda.OutOfMemoryError("Fixture GPU ran out of memory")
        return None, [{"update": 1, "time": 64., "loss": float("nan")}]

    monkeypatch.setattr(module, "fit_pilot", failed_fit)
    result = module.run_probe(tmp_path / "data", tmp_path / "probe.json")
    assert result["status"] == "failed" and "finite_loss" not in result
    assert result["peak_allocated_bytes"] == 1_500_000_000
    assert result["elapsed_seconds"] >= 0 and result["traceback"]
    assert json.loads((tmp_path / "probe.json").read_text()) == result


def test_probe_has_no_cpu_mode(tmp_path):
    module = _module()
    with pytest.raises(SystemExit) as error:
        module.main(["--dataset", str(tmp_path), "--output", str(tmp_path / "probe.json"),
                     "--device", "cpu"])
    assert error.value.code == 2 and not (tmp_path / "probe.json").exists()
