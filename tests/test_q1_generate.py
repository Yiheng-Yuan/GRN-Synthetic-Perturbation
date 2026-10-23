"""Q1 orchestration fixtures test I/O rules, not biological stationarity."""

import json
from pathlib import Path

import numpy as np
import pytest

from grn_experiment import q1_generate as batch
from grn_experiment.protocol import ReplicatedSnapshot
from grn_experiment.scmultisim import parameter_digest
from grn_experiment.scmultisim_observations import GeneratedSnapshot
from grn_experiment.scmultisim_stationary import StationarityResult
from grn_experiment.simulation import make_modular_network


def configuration():
    root = Path(__file__).parents[1]
    config = json.loads((root / "configs/q1-modular-64.json").read_text())
    return {**config, "cells_per_replicate": 2}


def fake_diagnostic(job):
    params, config, target = job
    return StationarityResult(True, 48. if target is None else 56., 56., {
        "schema_version": "scmultisim-stationarity-v1", "status": "diagnostic_passed",
        "parameter_sha256": parameter_digest(params), "target": target,
        "dose": 0. if target is None else config["dose"], "tau": config["tau"],
    })


class FakeSampler:
    def __init__(self, params, **kwargs):
        assert kwargs["save_production_jacobians"] is False
        self.shape = (3, kwargs["cells_per_replicate"], params.n_genes)

    def sample_with_truth(self, key):
        counts = np.ones(self.shape, dtype=np.int64)
        snapshot = ReplicatedSnapshot(counts, None if key.target is None else counts.copy())
        return GeneratedSnapshot(snapshot, {"treated_final_M": counts.astype(float)},
                                 {"duration": key.time, "fixture_not_a_stationarity_test": True})


def test_split_frozen_by_module_not_rna():
    network = make_modular_network(201)
    split = batch.registered_split(network, 20261023)
    assert split == batch.registered_split(network, 20261023)
    assert len(split["train"]) == 32 and len(split["held"]) == 16
    assert not set(split["train"]).intersection(split["held"])
    assert set(split["train"] + split["held"]) == set(network.regulators)
    for index, module in enumerate(network.modules):
        assert len(set(module).intersection(split["train"])) == (5 if index < 4 else 6)


@pytest.mark.parametrize("change", [
    {"phase": "blind"}, {"profile": "other"}, {"seed": True},
    {"tau": .1}, {"tau": float("nan")}, {"dose": .9},
    {"sampling_times": [.25, True, 2., 4., 8.]}, {"unknown": 1},
    {"cells_per_replicate": 0},
])
def test_bad_configuration_rejected_before_output(tmp_path, change):
    with pytest.raises(ValueError):
        batch.generate({**configuration(), **change}, tmp_path / "no-output", workers=1)
    assert not (tmp_path / "no-output").exists()


def test_whole_manifest_resume_and_learner_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "_diagnose", fake_diagnostic)
    monkeypatch.setattr(batch, "SnapshotSampler", FakeSampler)
    output = tmp_path / "data"
    manifest = batch.generate(configuration(), output, workers=1)
    assert manifest["status"] == "complete"
    assert len(manifest["conditions"]) == 289
    assert len(manifest["stationarity_diagnostics"]) == 49
    assert manifest["common_longterm_time"] == 56.
    assert manifest["total_cell_observations"] == 3462
    long_entries = [e for e in manifest["conditions"].values() if e.get("actual_sample_time") == 56.]
    assert len(long_entries) == 48
    assert all(e["diagnostic"]["passed"] for e in long_entries)
    from grn_experiment.q1_pilot_data import load_modular_q1, split_registered_targets
    data = load_modular_q1(output)
    train, held = split_registered_targets(data)
    assert len(train) == 192 and len(held) == 96
    assert data.baseline.shape == (3, 2, 64)
    assert all("private" not in name for name in data.provenance["verified_observation_sha256"])
    monkeypatch.setattr(batch, "_diagnose", lambda job: pytest.fail("Must not regenerate"))
    assert batch.generate(configuration(), output, workers=1) == manifest
    first = next(iter(manifest["conditions"].values()))
    (output / first["observation"]).write_bytes(b"corrupted fixture")
    with pytest.raises(ValueError, match="altered"):
        batch.generate(configuration(), output, workers=1)


def test_failed_stationarity_never_becomes_longterm_data(tmp_path, monkeypatch):
    def fail_one(job):
        result = fake_diagnostic(job)
        if job[2] is None:
            return StationarityResult(False, None, 128., result.report)
        return result
    monkeypatch.setattr(batch, "_diagnose", fail_one)
    output = tmp_path / "failed-data"
    manifest = batch.generate(configuration(), output, workers=1)
    assert manifest["status"] == "stationarity_failed"
    assert manifest["stationarity_failures"] == [None]
    assert manifest["conditions"] == {}
    assert not list((output / "observations").glob("**/*.npz"))
