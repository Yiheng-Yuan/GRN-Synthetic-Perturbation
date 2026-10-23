"""Read-only integrity audit of a completed modular-Q1 dataset and optional run.

This does not rerun stochastic trajectories, certify stationarity, infer a GRN,
or prove which files a training process accessed. It checks saved artifacts and
their declared contracts. Only the explicitly requested new JSON is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from grn_experiment.protocol import COMMON_BASELINE, Condition
from grn_experiment.q1_generate import GENERATOR_SOURCES, registered_split, validated_config
from grn_experiment.q1_pilot_data import _diagnostic, load_modular_q1
from grn_experiment.q1_pilot_evaluation import score_prediction
from grn_experiment.scmultisim import KineticParameters, development_parameters, parameter_digest
from grn_experiment.scmultisim_observations import key_id, key_record, stream_seed
from grn_experiment.simulation import make_modular_network, validate_network


MODELS = ("latent_neural_ode", "no_response", "direct_only_transfer")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file(root: Path, relative: str) -> Path:
    """Reject redirects rather than following a manifest outside its root."""
    _require(isinstance(relative, str) and bool(relative), "Missing registered file path")
    parts = PurePosixPath(relative)
    _require(not parts.is_absolute() and ".." not in parts.parts
             and str(parts) == relative and "\\" not in relative, "Unsafe registered path")
    path = root
    for part in parts.parts:
        path /= part
        _require(not path.is_symlink(), "Registered paths must not contain symlinks")
    _require(path.is_file() and path.resolve().is_relative_to(root), "Missing registered file")
    return path


def _json(path: Path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hashes(root: Path, records: dict) -> int:
    _require(isinstance(records, dict) and bool(records), "Missing registered checksums")
    for relative, expected in records.items():
        _require(_sha(_file(root, relative)) == expected, f"Checksum mismatch: {relative}")
    return len(records)


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        _require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ array")
        return {name: archive[name] for name in archive.files}


def _check_diagnostic(value, target, dose, params, config) -> float:
    burnin, report = _diagnostic(value, target, dose)
    _require(report["parameter_sha256"] == parameter_digest(params)
             and report["tau"] == config["tau"]
             and report.get("config") == config["stationarity"],
             "Diagnostic parameters, feedback grid or thresholds do not match")
    return burnin


def _check_snapshot(root, entry, key, params, config, baseline_burnin, seeds) -> tuple[int, int]:
    identifier, record = key_id(key), key_record(key)
    observation = f"observations/q1/{identifier}.npz"
    truth = f"private/truth/{identifier}.npz"
    provenance = f"private/provenance/{identifier}.json"
    _require(entry.get("condition") == record and entry.get("status") == "complete"
             and entry.get("observation") == observation
             and set(entry.get("files", {})) == {observation, truth, provenance},
             "Condition paths or metadata do not match the registered grid")
    count = _hashes(root, entry["files"])
    rna, states = _arrays(_file(root, observation)), _arrays(_file(root, truth))
    arms = ("treated",) if key == COMMON_BASELINE else ("treated", "control")
    _require(set(rna) == ({"treated"} if len(arms) == 1 else {"treated", "matched_control"}),
             "Unexpected RNA arrays")
    expected = {f"{arm}_{stage}_{species}" for arm in arms
                for stage in ("initial", "final") for species in ("G", "U", "M")}
    _require(set(states) == expected, "Missing or unexpected private joint states")
    shape = (3, config["cells_per_replicate"], params.n_genes)
    for name, value in states.items():
        _require(value.shape == shape and np.isfinite(value).all() and np.all(value >= 0),
                 f"Invalid finite nonnegative joint state: {name}")
        if name.endswith("_G"):
            _require(np.isin(value, (0, 1)).all(), "Promoter state must be binary")
    for arm in arms:
        value = rna["treated" if arm == "treated" else "matched_control"]
        _require(value.shape == shape and np.issubdtype(value.dtype, np.integer)
                 and np.all(value >= 0), "RNA must be a nonnegative integer cohort")
        _require(np.array_equal(value, np.floor(states[f"{arm}_final_M"])),
                 "Integer RNA does not equal floor(final continuous M)")
    provenance = _json(_file(root, provenance))
    _require(provenance.get("condition") == record and provenance.get("duration") == key.time
             and provenance.get("burnin") == baseline_burnin
             and provenance.get("tau") == config["tau"]
             and provenance.get("paired_cells_visible") is False
             and provenance.get("production_jacobians_saved") is False,
             "Snapshot duration, burn-in or observation contract differs")
    expected_streams = []
    for arm in arms:
        for replicate in range(3):
            row = {"arm": arm, "replicate": replicate}
            for kind in ("initial", "trajectory"):
                seed = str(stream_seed(config["seed"], record, arm, replicate, kind))
                _require(seed not in seeds, "A random stream is reused across cohorts")
                seeds.add(seed)
                row[f"{kind}_seed"] = seed
            expected_streams.append(row)
    _require(provenance.get("streams") == expected_streams, "Saved seeds fail exact recomputation")
    return count, len(arms)


def audit_dataset(directory: Path) -> tuple[dict, object]:
    """Inspect every registered completed artifact without simulating new cells."""
    root = Path(directory).resolve()
    manifest_path = _file(root, "manifest.json")
    manifest = _json(manifest_path)
    _require(manifest.get("status") == "complete", "Dataset is not complete; audit must wait")
    data = load_modular_q1(root)  # Strict public grid, hashes and actual-time checks.
    count = _hashes(root, manifest["private_files"])
    generator = _json(_file(root, "private/generator.json"))
    fingerprint = hashlib.sha256((json.dumps(generator, indent=2, sort_keys=True,
                                            allow_nan=False) + "\n").encode()).hexdigest()
    _require(fingerprint == manifest["fingerprint"], "Generator fingerprint mismatch")
    config = validated_config(generator["config"])
    params = KineticParameters.from_dict(generator["parameters"])
    network = make_modular_network(config["network_seed"])
    topology = validate_network(network, n_edges=128, require_regulator_propagation=True)
    _require(params.to_dict() == development_parameters(config["network_seed"], 64).to_dict()
             and np.array_equal(params.A, network.weights), "Saved parameters or graph differ from their seed")
    _require(generator.get("R_package_called") is False
             and generator.get("runtime_implementation") == "independent_python_scmultisim_derived_extension",
             "Generator implementation attribution differs")
    _require(generator.get("q1_split") == manifest["q1_split"] == registered_split(network, config["split_seed"])
             and generator.get("modules") == [list(m) for m in network.modules]
             and generator.get("isolated_genes") == list(network.isolated_genes),
             "Topology or target registration differs")
    _require(set(generator["source_sha256"]) == set(GENERATOR_SOURCES), "Incomplete generator source registration")
    for name, expected in generator["source_sha256"].items():
        relative = f"private/source_snapshot/{name}"
        _require(manifest["private_files"].get(relative) == expected
                 and _sha(_file(root, relative)) == expected, "Generator source archive mismatch")
    diagnostics = manifest["stationarity_diagnostics"]
    _require(set(diagnostics) == {str(q) for q in (None, *network.regulators)}, "Expected all 49 diagnostics")
    burnins = {q: _check_diagnostic(diagnostics[str(q)], q, 0. if q is None else .7, params, config)
               for q in (None, *network.regulators)}
    _require(manifest["baseline_stationarity"] == diagnostics["None"], "Baseline diagnostic mismatch")
    long_time = max(burnins[q] for q in network.regulators)
    _require(manifest["common_longterm_time"] == long_time, "Common long time differs from accepted diagnostics")
    grid = [COMMON_BASELINE, *(Condition(q, .7, t) for q in network.regulators
                               for t in (*config["sampling_times"], long_time))]
    _require(len(grid) == manifest["planned_timecourse_keys"] == 289
             and set(manifest["conditions"]) == {key_id(k) for k in grid}, "Incomplete or extra condition grid")
    seeds, arms = set(), 0
    for key in grid:
        entry = manifest["conditions"][key_id(key)]
        if key.time == long_time:
            _require(entry.get("diagnostic") == diagnostics[str(key.target)]
                     and entry.get("actual_sample_time") == long_time,
                     "Long-term condition diagnostic or actual time differs")
        verified, cohorts = _check_snapshot(root, entry, key, params, config, burnins[None], seeds)
        count += verified
        arms += cohorts
    _require(manifest["total_cell_observations"] == arms * 3 * config["cells_per_replicate"],
             "Cell observation count differs")
    report = {"status": "passed", "manifest_sha256": _sha(manifest_path),
              "conditions": len(grid), "verified_files": count, "rna_arrays": arms,
              "joint_state_arrays": arms * 6, "distinct_recomputed_seeds": len(seeds),
              "stationarity_diagnostics": len(diagnostics), "common_longterm_time": long_time,
              "parameter_sha256": parameter_digest(params), "n_edges": topology["n_edges"],
              "train_targets": manifest["q1_split"]["train"], "held_targets": manifest["q1_split"]["held"],
              "cell_observations": manifest["total_cell_observations"], "independent_networks": 1}
    return report, data


def _prediction_id(condition: Condition) -> str:
    return f"q{condition.target}_d{condition.dose}_t{condition.time}"


def audit_run(directory: Path, data, dataset_report: dict) -> dict:
    """Recompute stored scores; inspect declarations, not live training accesses."""
    root = Path(directory).resolve()
    hashes = _json(_file(root, "SHA256SUMS.json"))
    verified = _hashes(root, hashes)
    required = {"model.pt", "run_plan.json", "progress.jsonl", "history.json", "scores.json", "predictions.npz", "report.json"}
    _require(required <= set(hashes), "Missing required run checksums")
    plan, report = _json(_file(root, "run_plan.json")), _json(_file(root, "report.json"))
    _require(report.get("status") == "completed" and report.get("integration_check", {}).get("passed") is True,
             "Training or its numerical prediction check did not complete")
    _require(plan.get("mode") == "modular_fullcourse_target_holdout"
             and plan.get("fit_replicates") == plan.get("baseline_replicates") == [0, 1]
             and plan.get("test_replicates") == [0, 1, 2]
             and plan.get("fit_targets") == dataset_report["train_targets"]
             and plan.get("test_targets") == dataset_report["held_targets"]
             and plan.get("checkpoint_selection") == "last fixed-budget update; no test-based selection"
             and report.get("held_out_responses_and_controls_used_for_fitting") is False,
             "Training/test separation declarations differ")
    _require(plan.get("input_provenance", {}).get("manifest_sha256") == dataset_report["manifest_sha256"],
             "Run is linked to another input dataset")
    for name in ("config", "fit_targets", "test_targets", "source_sha256", "fit_conditions", "test_conditions"):
        _require(plan.get(name) == report.get(name), f"Run plan/report differ: {name}")
    _require(plan["fit_conditions"] == 192 and plan["test_conditions"] == 96, "Wrong fit/test condition counts")
    source_hashes = plan["source_sha256"]
    _require(set(source_hashes) == {"q1_pilot.py", "q1_pilot_data.py", "q1_pilot_evaluation.py", "protocol.py"},
             "Missing training source registration")
    for name, expected in source_hashes.items():
        _require(_sha(_file(root, f"source_snapshot/grn_experiment/{name}")) == expected,
                 "Training source archive mismatch")
    history = _json(_file(root, "history.json"))
    progress = [json.loads(line) for line in _file(root, "progress.jsonl").read_text().splitlines()]
    config = plan["config"]
    _require(type(config.get("updates")) is int and config["updates"] > 0
             and len(history) == config["updates"] and history == progress, "History/progress update count differs")
    times = sorted({key.time for key in data.observations})
    _require(config["fit_times"] == times, "Training times differ from the complete observed grid")
    for index, row in enumerate(history):
        _require(row.get("update") == index + 1 and row.get("time") == times[index % len(times)]
                 and all(np.isfinite(row.get(name, np.nan)) for name in
                         ("loss", "distribution", "control", "effect", "target_effect")),
                 "Nonfinite, unbalanced or misnumbered training history")
        expected = row["distribution"] + row["control"] + config["effect_weight"] * row["effect"] + config["target_weight"] * row["target_effect"]
        _require(np.isclose(row["loss"], expected, rtol=2e-6, atol=1e-7), "Training loss does not match its components")
    predictions = _arrays(_file(root, "predictions.npz"))
    expected = {f"{model}_{_prediction_id(key)}_{arm}" for key in data.observations
                for model in MODELS for arm in ("treated", "control")}
    _require(set(predictions) == expected and len(expected) == 1728, "Prediction grid differs")
    for value in predictions.values():
        _require(value.ndim == 2 and value.shape[1] == 64 and value.shape[0] > 0
                 and np.isfinite(value).all() and np.all(value >= 0), "Invalid predicted RNA population")
    rows = _json(_file(root, "scores.json"))
    expected_rows, maximum_error = {}, 0.
    for key, snapshot in data.observations.items():
        held = key.target in dataset_report["held_targets"]
        for model in MODELS:
            a = predictions[f"{model}_{_prediction_id(key)}_treated"]
            b = predictions[f"{model}_{_prediction_id(key)}_control"]
            for replicate in (range(3) if held else (2,)):
                identity = (model, key.target, key.dose, key.time, replicate)
                expected_rows[identity] = {"model": model, "split": "held_target" if held else "fit_condition_check",
                    "stage": "long_term" if key.time > 8 else "timecourse",
                    **score_prediction(a, b, key, snapshot, replicate=replicate)}
    _require(len(rows) == len(expected_rows) == 1440, "Score count differs")
    seen = set()
    for row in rows:
        identity = tuple(row.get(name) for name in ("model", "target", "dose", "time", "replicate"))
        _require(identity in expected_rows and identity not in seen, "Duplicate or unexpected score row")
        seen.add(identity)
        recomputed = expected_rows[identity]
        _require(set(row) == set(recomputed), "Score fields differ")
        for name, value in recomputed.items():
            if isinstance(value, (int, float)):
                error = abs(row[name] - value)
                maximum_error = max(maximum_error, error)
                _require(np.isclose(row[name], value, atol=1e-9, rtol=1e-9), f"Score differs: {name}")
            else:
                _require(row[name] == value, f"Score metadata differs: {name}")
    return {"status": "passed", "verified_top_level_files": verified,
            "verified_training_sources": len(source_hashes), "updates": len(history),
            "updates_by_time": dict(Counter(str(row["time"]) for row in history)),
            "prediction_arrays": len(predictions), "recomputed_score_rows": len(rows),
            "max_score_absolute_difference": maximum_error,
            "score_check": "recomputed using published score_prediction; not a second metric implementation",
            "access_check": "saved split/replicate declarations checked; not runtime file-access monitoring"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(p.resolve()) for p in (args.dataset, args.run) if p is not None):
        parser.error("output must be a new file outside the read-only input directories")
    dataset, data = audit_dataset(args.dataset)
    report = {"schema": "q1-modular-artifact-audit-v1", "status": "passed", "dataset": dataset,
              "scope": "artifact integrity only; not proof of biological realism, stationarity, propagation signal or learnability",
              "audit_script_sha256": _sha(Path(__file__))}
    if args.run is not None:
        report["run"] = audit_run(args.run, data, dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
