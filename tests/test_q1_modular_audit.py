"""Small synthetic artifact fixtures; no dynamics simulation or model training."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from grn_experiment import q1_generate as batch
from grn_experiment.protocol import ReplicatedSnapshot
from grn_experiment.scmultisim import parameter_digest
from grn_experiment.scmultisim_observations import GeneratedSnapshot, key_record, stream_seed
from grn_experiment.scmultisim_stationary import StationarityResult


script = Path(__file__).parents[1] / "tools/scmultisim/audit_q1_modular.py"
spec = importlib.util.spec_from_file_location("audit_q1_modular", script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _diagnostic(job):
    params, config, target = job
    return StationarityResult(True, 48., 48., {
        "schema_version": "scmultisim-stationarity-v1", "status": "diagnostic_passed",
        "parameter_sha256": parameter_digest(params), "target": target,
        "dose": 0. if target is None else .7, "tau": config["tau"],
        "config": config["stationarity"], "fixture_not_scientific_evidence": True})


class SamplerFixture:
    def __init__(self, params, *, seed, tau, baseline_stationarity, cells_per_replicate, **kwargs):
        self.seed, self.tau, self.burnin = seed, tau, baseline_stationarity.burnin
        self.shape = (3, cells_per_replicate, params.n_genes)

    def sample_with_truth(self, key):
        truth, streams, observations = {}, [], {}
        for arm in (("treated",) if key.target is None else ("treated", "control")):
            M = np.full(self.shape, 5.25)
            if arm == "treated" and key.target is not None:
                M[..., key.target] = 3.75
            observations[arm] = np.floor(M).astype(np.int64)
            for stage in ("initial", "final"):
                truth[f"{arm}_{stage}_G"] = np.ones(self.shape, dtype=np.uint8)
                truth[f"{arm}_{stage}_U"] = np.ones(self.shape)
                truth[f"{arm}_{stage}_M"] = M.copy()
            for replicate in range(3):
                streams.append({"arm": arm, "replicate": replicate,
                    **{f"{kind}_seed": str(stream_seed(self.seed, key_record(key), arm, replicate, kind))
                       for kind in ("initial", "trajectory")}})
        return GeneratedSnapshot(ReplicatedSnapshot(observations["treated"], observations.get("control")), truth,
            {"condition": key_record(key), "duration": key.time, "burnin": self.burnin, "tau": self.tau,
             "paired_cells_visible": False, "production_jacobians_saved": False, "streams": streams})


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).parents[1] / "configs/q1-modular-64.json").read_text())
    config["cells_per_replicate"] = 2
    monkeypatch.setattr(batch, "_diagnose", _diagnostic)
    monkeypatch.setattr(batch, "SnapshotSampler", SamplerFixture)
    directory = tmp_path / "data"
    batch.generate(config, directory, workers=1)
    return directory


def test_audit_all_artifacts_and_no_overwrite(dataset, tmp_path):
    before = audit._sha(dataset / "manifest.json")
    report, _ = audit.audit_dataset(dataset)
    assert report["conditions"] == 289
    assert report["verified_files"] == 875
    assert report["distinct_recomputed_seeds"] == 3462
    assert report["rna_arrays"] == 577
    output = tmp_path / "audit.json"
    assert audit.main(["--dataset", str(dataset), "--output", str(output)]) == 0
    assert audit._sha(dataset / "manifest.json") == before
    with pytest.raises(SystemExit):
        audit.main(["--dataset", str(dataset), "--output", str(output)])
    with pytest.raises(SystemExit):
        audit.main(["--dataset", str(dataset), "--output", str(dataset / "audit.json")])


@pytest.mark.parametrize("mutation", ["hash", "floor", "seed", "joint_state", "unfinished"])
def test_dataset_corruption_rejected(dataset, mutation):
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "unfinished":
        manifest["status"] = "generating"
    else:
        entry = next(iter(manifest["conditions"].values()))
        if mutation == "hash":
            (dataset / entry["observation"]).write_bytes(b"broken")
        else:
            relative = next(name for name in entry["files"] if
                            name.startswith("private/provenance" if mutation == "seed" else "private/truth"))
            path = dataset / relative
            if mutation == "seed":
                value = json.loads(path.read_text())
                value["streams"][0]["initial_seed"] = "42"
                _json(path, value)
            else:
                arrays = audit._arrays(path)
                arrays["treated_final_M"][0, 0, 0] = 20. if mutation == "floor" else -1.
                np.savez_compressed(path, **arrays)
            entry["files"][relative] = audit._sha(path)
    _json(manifest_path, manifest)
    with pytest.raises(ValueError):
        audit.audit_dataset(dataset)


def _run_fixture(directory, data, dataset_report):
    directory.mkdir()
    source_hashes = {}
    package = Path(__file__).parents[1] / "src/grn_experiment"
    for name in ("q1_pilot.py", "q1_pilot_data.py", "q1_pilot_evaluation.py", "protocol.py"):
        target = directory / "source_snapshot/grn_experiment" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((package / name).read_bytes())
        source_hashes[name] = audit._sha(target)
    times = sorted({key.time for key in data.observations})
    plan = {"mode": "modular_fullcourse_target_holdout", "fit_replicates": [0, 1],
        "baseline_replicates": [0, 1], "test_replicates": [0, 1, 2],
        "fit_targets": dataset_report["train_targets"], "test_targets": dataset_report["held_targets"],
        "fit_conditions": 192, "test_conditions": 96,
        "checkpoint_selection": "last fixed-budget update; no test-based selection",
        "input_provenance": {"manifest_sha256": dataset_report["manifest_sha256"]},
        "source_sha256": source_hashes,
        "config": {"updates": 6, "fit_times": times, "effect_weight": 1., "target_weight": 0.}}
    history = [{"update": i + 1, "time": t, "loss": 0., "distribution": 0., "control": 0.,
                "effect": 0., "target_effect": 0.} for i, t in enumerate(times)]
    predictions, scores = {}, []
    for key, snapshot in data.observations.items():
        held = key.target in dataset_report["held_targets"]
        for model in audit.MODELS:
            a, b = snapshot.treated[0], snapshot.matched_control[0]
            for arm, value in (("treated", a), ("control", b)):
                predictions[f"{model}_{audit._prediction_id(key)}_{arm}"] = value
            for replicate in (range(3) if held else (2,)):
                scores.append({"model": model, "split": "held_target" if held else "fit_condition_check",
                    "stage": "long_term" if key.time > 8 else "timecourse",
                    **audit.score_prediction(a, b, key, snapshot, replicate=replicate)})
    _json(directory / "run_plan.json", plan)
    _json(directory / "history.json", history)
    (directory / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    _json(directory / "scores.json", scores)
    _json(directory / "report.json", {**plan, "status": "completed", "integration_check": {"passed": True},
                                      "held_out_responses_and_controls_used_for_fitting": False})
    (directory / "model.pt").write_bytes(b"fixture_not_weights")
    np.savez_compressed(directory / "predictions.npz", **predictions)
    _json(directory / "SHA256SUMS.json", {p.name: audit._sha(p) for p in directory.iterdir() if p.is_file()})


def test_run_scores_sources_history_and_replicate_declarations(dataset, tmp_path):
    dataset_report, data = audit.audit_dataset(dataset)
    directory = tmp_path / "run"
    _run_fixture(directory, data, dataset_report)
    result = audit.audit_run(directory, data, dataset_report)
    assert result["prediction_arrays"] == 1728
    assert result["recomputed_score_rows"] == 1440
    assert result["updates"] == 6
    assert result["max_score_absolute_difference"] == 0.
    # Re-register modified bytes so this checks the contract, not only its hash.
    plan = json.loads((directory / "run_plan.json").read_text())
    plan["fit_replicates"] = [0, 1, 2]
    _json(directory / "run_plan.json", plan)
    hashes = json.loads((directory / "SHA256SUMS.json").read_text())
    hashes["run_plan.json"] = audit._sha(directory / "run_plan.json")
    _json(directory / "SHA256SUMS.json", hashes)
    with pytest.raises(ValueError, match="separation"):
        audit.audit_run(directory, data, dataset_report)
