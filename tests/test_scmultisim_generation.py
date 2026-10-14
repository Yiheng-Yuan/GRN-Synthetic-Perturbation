"""Batch I/O tests with a short diagnostic stub, not convergence evidence."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from grn_experiment import generate_scmultisim as batch
from grn_experiment.protocol import COMMON_BASELINE, Condition
from grn_experiment.scmultisim import parameter_digest


def config():
    return {"phase": "development", "seed": 1, "network_seed": 101, "split_seed": 1,
            "cells_per_replicate": 2, "tau": 0.125, "scope": "initial", "stationary": False,
            "stationarity": {}}


def diagnostic(passed=True, params=None, tau=.125, target=None, dose=0.):
    report = {"parameter_sha256": parameter_digest(params) if params else None,
              "tau": tau, "target": target, "dose": dose}
    return SimpleNamespace(passed=passed, burnin=0.125 if passed else None,
                           report=report,
                           to_dict=lambda: {"passed": passed, "burnin": 0.125 if passed else None})


def use_short_fixture(monkeypatch):
    monkeypatch.setattr(batch, "assess_stationarity", lambda params, **kwargs:
                        diagnostic(params=params, tau=kwargs["tau"],
                                   target=kwargs.get("target"), dose=kwargs.get("dose", 0.)))
    monkeypatch.setattr(batch, "planned_conditions", lambda split, scope: (COMMON_BASELINE, Condition(split.initial[0], 0.7, 0.25)))


def test_generation_writes_floor_counts_truth_and_resumes_without_sampling(tmp_path, monkeypatch):
    use_short_fixture(monkeypatch)
    output = tmp_path / "data"
    first = batch.generate(config(), output)
    assert first["status"] == "complete"
    assert len(first["conditions"]) == 2
    assert (output / "private/generator.json").exists()
    for entry in first["conditions"].values():
        with np.load(output / entry["observation"], allow_pickle=False) as values:
            assert values["treated"].dtype == np.int64
            assert values["treated"].shape == (3, 2, 24)
            assert set(values.files) <= {"treated", "matched_control"}
    def no_work(*args, **kwargs):
        raise AssertionError("A complete run must not resimulate")
    monkeypatch.setattr(batch, "assess_stationarity", no_work)
    assert batch.generate(config(), output) == first


def test_resume_rejects_altered_data_and_changed_config(tmp_path, monkeypatch):
    use_short_fixture(monkeypatch)
    output = tmp_path / "data"
    manifest = batch.generate(config(), output)
    changed = {**config(), "seed": 2}
    with pytest.raises(ValueError, match="changed"):
        batch.generate(changed, output)
    entry = next(iter(manifest["conditions"].values()))
    (output / entry["observation"]).write_bytes(b"corrupted test fixture")
    with pytest.raises(ValueError, match="Missing or altered"):
        batch.generate(config(), output)


def test_baseline_failure_reports_and_generates_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "assess_stationarity", lambda *args, **kwargs: diagnostic(False))
    output = tmp_path / "data"
    result = batch.generate(config(), output)
    assert result["status"] == "baseline_stationarity_failed"
    assert result["conditions"] == {}
    assert not list(output.rglob("*.npz"))
    assert json.loads((output / "manifest.json").read_text())["baseline_stationarity"]["passed"] is False


def test_formal_and_unknown_config_rejected_before_creating_output(tmp_path):
    for changed in ({**config(), "phase": "blind"}, {**config(), "unexpected": True}):
        output = tmp_path / "not-created"
        with pytest.raises(ValueError):
            batch.generate(changed, output)
        assert not output.exists()


def test_interrupted_condition_resumes_completed_files(tmp_path, monkeypatch):
    use_short_fixture(monkeypatch)
    original = batch._write_snapshot
    calls = []
    def fail_second(output, key, split, generated):
        calls.append(key)
        if len(calls) == 2:
            raise RuntimeError("synthetic interruption")
        return original(output, key, split, generated)
    monkeypatch.setattr(batch, "_write_snapshot", fail_second)
    output = tmp_path / "data"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        batch.generate(config(), output)
    partial = json.loads((output / "manifest.json").read_text())
    assert partial["status"] == "interrupted_or_failed"
    assert len(partial["conditions"]) == 1
    monkeypatch.setattr(batch, "_write_snapshot", original)
    completed = batch.generate(config(), output)
    assert completed["status"] == "complete" and len(completed["conditions"]) == 2
    assert completed["conditions"][next(iter(partial["conditions"]))] == next(iter(partial["conditions"].values()))
