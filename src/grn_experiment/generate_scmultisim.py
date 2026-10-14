"""Generate one development dataset; this module never trains a model.

The output directory is a generator/scorer vault, not a learner mount. Use
``load_blind_store`` for the protocol's staged observation interface. Formal
blind-network generation remains disabled until scientific calibration.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

from .protocol import (
    COMMON_BASELINE, BlindStore, Condition, SteadyKey, ReplicatedSnapshot, Split,
    make_split, initial_grid, candidate_grid, validation_grid,
    required_test_conditions, stationary_grid,
)
from .scmultisim import KineticParameters, development_parameters, KERNEL_VERSION, SOURCE_COMMIT
from .scmultisim_observations import SnapshotSampler, key_id, key_record, stream_seed
from .scmultisim_stationary import StationarityConfig, assess_stationarity, calibrate_common_K
from .selection import uniform_sample


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest() -> str:
    """Include implementation and protocol sources, including uncommitted edits."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    _atomic(path, lambda stream: stream.write(_json_bytes(value)))


def _atomic_npz(path: Path, arrays: dict) -> None:
    _atomic(path, lambda stream: np.savez_compressed(stream, **arrays))


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _directory_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".generation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is generating this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _check_files(output: Path, files: dict[str, str]) -> None:
    for relative, checksum in files.items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Manifest path escapes the output directory")
        if not path.is_file() or _digest(path) != checksum:
            raise ValueError(f"Missing or altered output: {relative}; use a new output directory")


def planned_conditions(split: Split, scope: str) -> tuple[Condition, ...]:
    keys = [COMMON_BASELINE, *initial_grid(split)]
    if scope in ("full", "pool"):
        keys.extend(candidate_grid(split) if scope == "pool" else uniform_sample(candidate_grid(split)))
        keys.extend(validation_grid(split))
        keys.extend(required_test_conditions(split))
    elif scope != "initial":
        raise ValueError("scope must be initial, full, or pool")
    return tuple(dict.fromkeys(keys))


def _visibility(key: Condition | SteadyKey, split: Split) -> str:
    if isinstance(key, SteadyKey):
        return "sealed/stationary"
    if key == COMMON_BASELINE or key in initial_grid(split):
        return "observations/initial"
    if key in candidate_grid(split):
        return "sealed/candidates"
    if key in validation_grid(split):
        return "sealed/validation"
    return "sealed/final"


def _write_snapshot(output: Path, key, split, generated) -> dict:
    identifier = key_id(key)
    observation_path = Path(_visibility(key, split)) / f"{identifier}.npz"
    truth_path = Path("private/truth") / f"{identifier}.npz"
    provenance_path = Path("private/provenance") / f"{identifier}.json"
    snapshot = generated.snapshot
    arrays = {"treated": snapshot.treated.astype(np.int64)}
    if snapshot.matched_control is not None:
        arrays["matched_control"] = snapshot.matched_control.astype(np.int64)
    _atomic_npz(output / observation_path, arrays)
    _atomic_npz(output / truth_path, generated.truth)
    _atomic_json(output / provenance_path, generated.provenance)
    files = {str(path): _digest(output / path) for path in (observation_path, truth_path, provenance_path)}
    return {"condition": key_record(key), "status": "complete", "observation": str(observation_path), "files": files}


def _validated_config(config: dict) -> dict:
    allowed = {"phase", "seed", "network_seed", "split_seed", "cells_per_replicate", "tau", "scope", "stationary", "stationarity", "parameter_file"}
    if set(config) - allowed:
        raise ValueError(f"Unknown generation settings: {sorted(set(config) - allowed)}")
    if config.get("phase") != "development":
        raise ValueError("Only phase=development is implemented; formal blind generation requires calibration")
    required = allowed - {"parameter_file"}
    if required - set(config):
        raise ValueError(f"Missing settings: {sorted(required - set(config))}")
    for key in ("seed", "network_seed", "split_seed"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    cells = config["cells_per_replicate"]
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
        raise ValueError("cells_per_replicate must be a positive integer")
    if isinstance(config["tau"], bool) or not np.isfinite(config["tau"]) or config["tau"] <= 0:
        raise ValueError("tau must be finite and positive")
    if config["scope"] not in ("initial", "full", "pool"):
        raise ValueError("scope must be initial, full, or pool")
    if not isinstance(config["stationary"], bool):
        raise ValueError("stationary must be boolean")
    if config["scope"] == "initial" and config["stationary"]:
        raise ValueError("initial scope does not include final stationary scoring")
    StationarityConfig(**config["stationarity"])
    return config


def generate(config: dict, output: Path, *, config_directory: Path = Path(".")) -> dict:
    """Generate/resume complete condition artifacts, preserving valid files.

    A restart recomputes deterministic stationarity diagnostics if interrupted,
    then skips completed, checksum-verified condition bundles. A condition
    interrupted before its manifest commit is regenerated with its same seeds.
    """
    config = _validated_config(dict(config))
    if config.get("parameter_file"):
        params = KineticParameters.from_dict(json.loads((config_directory / config["parameter_file"]).read_text()))
    else:
        params = development_parameters(config["network_seed"], n_genes=24)
        # One shared development reference, independent of the selected graph's
        # observed ctrl endpoints. This candidate does not freeze the main plan.
        reference = calibrate_common_K([development_parameters(seed) for seed in (101, 102, 103, 104)])
        values = params.to_dict()
        values["K"] = reference["K"]
        params = KineticParameters.from_dict(values)
    if params.n_genes != 24 or len(params.regulators) != 16:
        raise ValueError("Protocol generation requires 24 genes and 16 regulators")
    indegree, outdegree = (params.A != 0).sum(axis=1), (params.A != 0).sum(axis=0)
    nonregulators = sorted(set(range(24)) - set(params.regulators))
    if (np.count_nonzero(params.A) != 40 or np.any(indegree > 3) or
            np.any(outdegree > 5) or np.any(indegree[nonregulators] == 0)):
        raise ValueError("Protocol network must have 40 edges and meet degree/coverage constraints")
    split = make_split(params.regulators, config["split_seed"])
    conditions = planned_conditions(split, config["scope"])
    stationary_keys = stationary_grid(split) if config["stationary"] else ()
    provenance = {"config": config, "parameters": params.to_dict(), "source_sha256": source_digest(),
                  "python": platform.python_version(), "numpy": np.__version__,
                  "kernel_version": KERNEL_VERSION, "mechanism_reference_commit": SOURCE_COMMIT,
                  "runtime_implementation": "independent_python_scmultisim_derived_extension",
                  "R_package_called": False}
    if not config.get("parameter_file"):
        provenance["development_K_reference"] = reference
    fingerprint = hashlib.sha256(_json_bytes(provenance)).hexdigest()
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    with _directory_lock(output):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("Configuration, parameters, source, or runtime changed; use a new output directory")
            _check_files(output, manifest.get("private_files", {}))
            for entry in manifest["conditions"].values():
                _check_files(output, entry.get("files", {}))
            if manifest["status"] in ("complete", "complete_with_stationarity_failures"):
                return manifest
        else:
            if any(path.name != ".generation.lock" for path in output.iterdir()):
                raise ValueError("Output directory contains unmanaged files; use a new directory")
            manifest = {"schema": "scmultisim-development-v1", "fingerprint": fingerprint,
                        "status": "running", "formal_calibration": False,
                        "scope": config["scope"], "split": {name: list(values) for name, values in asdict(split).items()},
                        "cells_per_replicate": config["cells_per_replicate"], "n_genes": params.n_genes,
                        "planned_timecourse_keys": len(conditions), "planned_stationary_keys": len(stationary_keys),
                        "conditions": {}, "stationarity_failures": [], "private_files": {},
                        "warning": "Generator/scorer vault; do not mount this directory in learner processes."}
            _atomic_json(manifest_path, manifest)
        provenance_path = output / "private/generator.json"
        _atomic_json(provenance_path, provenance)
        manifest["private_files"]["private/generator.json"] = _digest(provenance_path)
        _atomic_json(manifest_path, manifest)
        try:
            diagnostic_config = StationarityConfig(**config["stationarity"])
            print("Assessing the unperturbed initial distribution", flush=True)
            baseline = assess_stationarity(params, seed=stream_seed(config["seed"], "baseline_diagnostic"),
                                           tau=config["tau"], config=diagnostic_config)
            manifest["baseline_stationarity"] = baseline.to_dict()
            _atomic_json(manifest_path, manifest)
            if not baseline.passed:
                manifest["status"] = "baseline_stationarity_failed"
                _atomic_json(manifest_path, manifest)
                return manifest
            sampler = SnapshotSampler(params, seed=config["seed"], tau=config["tau"],
                                      baseline_stationarity=baseline, cells_per_replicate=config["cells_per_replicate"])
            for index, key in enumerate(conditions, 1):
                identifier = key_id(key)
                if identifier in manifest["conditions"]:
                    continue
                print(f"Generating timecourse {index}/{len(conditions)}: {key_record(key)}", flush=True)
                manifest["conditions"][identifier] = _write_snapshot(output, key, split, sampler.sample_with_truth(key))
                _atomic_json(manifest_path, manifest)
            for index, key in enumerate(stationary_keys, 1):
                identifier = key_id(key)
                if identifier in manifest["conditions"]:
                    continue
                print(f"Generating stationary {index}/{len(stationary_keys)}: {key_record(key)}", flush=True)
                result = assess_stationarity(params, seed=stream_seed(config["seed"], "stationary_diagnostic", key_record(key)),
                                             tau=config["tau"], config=diagnostic_config, target=key.target, dose=key.dose)
                if not result.passed:
                    manifest["conditions"][identifier] = {"condition": key_record(key), "status": "stationarity_failed",
                                                          "diagnostic": result.to_dict()}
                    manifest["stationarity_failures"].append(key_record(key))
                else:
                    generated = sampler.sample_with_truth(key, stationary_result=result)
                    entry = _write_snapshot(output, key, split, generated)
                    entry["diagnostic"] = result.to_dict()
                    manifest["conditions"][identifier] = entry
                _atomic_json(manifest_path, manifest)
            manifest["status"] = "complete_with_stationarity_failures" if manifest["stationarity_failures"] else "complete"
            manifest.pop("error", None)
            _atomic_json(manifest_path, manifest)
            return manifest
        except Exception as exc:
            manifest["status"] = "interrupted_or_failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            _atomic_json(manifest_path, manifest)
            raise


def load_blind_store(output: Path) -> BlindStore:
    """Scorer-side reconstruction; call only after generation finishes.

    Raw paths and truth are never returned through this API. Filesystem
    permissions alone do not isolate learners sharing the same OS account.
    """
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("A complete dataset is required; failed stationary outcomes cannot be substituted")
    split = Split(**{name: tuple(values) for name, values in manifest["split"].items()})
    observations, stationary = {}, {}
    for entry in manifest["conditions"].values():
        _check_files(output, entry["files"])
        record = entry["condition"]
        with np.load(output / entry["observation"], allow_pickle=False) as arrays:
            snapshot = ReplicatedSnapshot(arrays["treated"], arrays["matched_control"] if "matched_control" in arrays else None)
        if record["kind"] == "timecourse":
            observations[Condition(record["target"], record["dose"], record["time"])] = snapshot
        else:
            stationary[SteadyKey(record["target"], record["dose"])] = snapshot
    return BlindStore(split, observations, n_genes=manifest["n_genes"],
                      cells_per_replicate=manifest["cells_per_replicate"], stationary_observations=stationary or None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        manifest = generate(config, args.output, config_directory=args.config.resolve().parent)
        print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()),
                          "completed_conditions": sum(x["status"] == "complete" for x in manifest["conditions"].values()),
                          "stationarity_failures": len(manifest["stationarity_failures"])}), flush=True)
        return {"complete": 0, "baseline_stationarity_failed": 2, "complete_with_stationarity_failures": 3}[manifest["status"]]
    except Exception as exc:
        print(f"Generation failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
