"""Generate the sparse 64-gene Q1 development dataset, never fit a model.

This is orchestration around the existing scMultiSim-derived Python kernel,
not another dynamics implementation. Every observed cohort is independent;
the common long sampling time is determined by all predeclared diagnostics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np

from .generate_scmultisim import (_atomic, _atomic_json, _atomic_npz, _check_files,
                                  _digest, _directory_lock, _json_bytes)
from .protocol import COMMON_BASELINE, Condition
from .scmultisim import KERNEL_VERSION, SOURCE_COMMIT, development_parameters
from .scmultisim_observations import SnapshotSampler, key_id, key_record, stream_seed
from .scmultisim_stationary import StationarityConfig, assess_stationarity
from .simulation import make_modular_network, validate_network


SCHEMA = "scmultisim-q1-modular-v1"
GENERATOR_SOURCES = ("q1_generate.py", "generate_scmultisim.py", "simulation.py",
                     "scmultisim.py", "scmultisim_observations.py",
                     "scmultisim_stationary.py", "protocol.py")


def validated_config(config: dict) -> dict:
    """Fail before creating output for unsupported or ambiguous experiments."""
    expected = {"phase", "profile", "seed", "network_seed", "split_seed",
                "cells_per_replicate", "tau", "dose", "sampling_times", "stationarity"}
    if set(config) != expected:
        raise ValueError("Q1 generation requires exactly the documented configuration fields")
    if config["phase"] != "development" or config["profile"] != "q1-modular-64-v1":
        raise ValueError("Only the q1-modular-64-v1 development profile is implemented")
    for name in ("seed", "network_seed", "split_seed", "cells_per_replicate"):
        if type(config[name]) is not int or config[name] < (1 if name == "cells_per_replicate" else 0):
            raise ValueError(f"Invalid integer {name}")
    if (type(config["tau"]) not in (int, float) or not np.isfinite(config["tau"])
            or not 0 < config["tau"] <= 1 / 32):
        raise ValueError("tau must be positive and at most 1/32")
    if type(config["dose"]) not in (int, float) or config["dose"] != .7:
        raise ValueError("The registered single-dose comparison uses dose 0.7")
    if (config["sampling_times"] != [.25, 1., 2., 4., 8.]
            or any(type(t) not in (int, float) for t in config["sampling_times"])):
        raise ValueError("The registered transient sampling times are .25/1/2/4/8")
    StationarityConfig(**config["stationarity"])
    return dict(config)


def registered_split(network, seed: int) -> dict:
    """Stratify by the predeclared modules, never by observed RNA responses.

    Module identities remain generator/scorer information. Only the resulting
    target lists are exposed to the learner, not a true connectivity mask.
    """
    rng = np.random.default_rng(seed)
    train, held = [], []
    for index, module in enumerate(network.modules):
        targets = sorted(set(module).intersection(network.regulators))
        if len(targets) != 8:
            raise ValueError("Each module must contain eight regulators")
        order = rng.permutation(targets).tolist()
        count = 5 if index < 4 else 6
        train.extend(order[:count])
        held.extend(order[count:])
    if len(train) != 32 or len(held) != 16:
        raise ValueError("Expected the predeclared 32/16 target split")
    return {"train": sorted(train), "held": sorted(held),
            "registration": "topology_before_rna"}


def _diagnose(job):
    params, config, target = job
    dose = 0. if target is None else config["dose"]
    seed = stream_seed(config["seed"], "q1_stationarity", target, dose)
    return assess_stationarity(params, seed=seed, tau=config["tau"], target=target,
                               dose=dose, config=StationarityConfig(**config["stationarity"]))


def _sample(job):
    output, params, config, baseline, key, diagnostic = job
    sampler = SnapshotSampler(params, seed=config["seed"], tau=config["tau"],
                              baseline_stationarity=baseline,
                              cells_per_replicate=config["cells_per_replicate"],
                              save_production_jacobians=False)
    generated = sampler.sample_with_truth(key)
    identifier = key_id(key)
    observation = f"observations/q1/{identifier}.npz"
    truth = f"private/truth/{identifier}.npz"
    provenance = f"private/provenance/{identifier}.json"
    arrays = {"treated": generated.snapshot.treated.astype(np.int64)}
    if generated.snapshot.matched_control is not None:
        arrays["matched_control"] = generated.snapshot.matched_control.astype(np.int64)
    _atomic_npz(output / observation, arrays)
    _atomic_npz(output / truth, generated.truth)
    _atomic_json(output / provenance, generated.provenance)
    entry = {"condition": key_record(key), "status": "complete", "observation": observation,
             "files": {name: _digest(output / name) for name in (observation, truth, provenance)}}
    if diagnostic is not None:
        entry.update(diagnostic=diagnostic, actual_sample_time=key.time)
    return identifier, entry


def _run_jobs(function, jobs, workers):
    """Bounded process parallelism; task seeds never depend on worker order."""
    if workers == 1:
        yield from map(function, jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            yield from pool.map(function, jobs, chunksize=1)


def generate(config: dict, output: Path, *, workers: int = 3) -> dict:
    """Resume only unchanged, checksum-verified artifacts; preserve all failures."""
    config = validated_config(config)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer from 1 through 8")
    network = make_modular_network(config["network_seed"])
    topology = validate_network(network, n_edges=128, require_motifs=True,
                                require_regulator_propagation=True)
    params = development_parameters(config["network_seed"], n_genes=64)
    if not np.array_equal(params.A, network.weights):
        raise ValueError("Parameter graph must match the registered topology")
    split = registered_split(network, config["split_seed"])
    sources = {name: _digest(Path(__file__).with_name(name)) for name in GENERATOR_SOURCES}
    provenance = {"config": config, "parameters": params.to_dict(), "source_sha256": sources,
                  "python": platform.python_version(), "numpy": np.__version__,
                  "kernel_version": KERNEL_VERSION, "mechanism_reference_commit": SOURCE_COMMIT,
                  "runtime_implementation": "independent_python_scmultisim_derived_extension",
                  "R_package_called": False, "K_reference": "fixed_24_gene_development_reference",
                  "modules": [list(module) for module in network.modules],
                  "isolated_genes": list(network.isolated_genes), "q1_split": split,
                  "truth_storage": "full_G_U_M_states_no_per_cell_dense_jacobians"}
    fingerprint = hashlib.sha256(_json_bytes(provenance)).hexdigest()
    output = Path(output).resolve()
    started = time.monotonic()
    with _directory_lock(output):
        path = output / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("Configuration, generator source or runtime changed; use a new output directory")
            _check_files(output, manifest.get("private_files", {}))
            for entry in manifest["conditions"].values():
                _check_files(output, entry["files"])
            if manifest["status"] == "complete":
                return manifest
        else:
            if any(item.name != ".generation.lock" for item in output.iterdir()):
                raise ValueError("Output has unmanaged files; use a new directory")
            manifest = {"schema": SCHEMA, "phase": "development", "formal_calibration": False,
                        "status": "running_diagnostics", "fingerprint": fingerprint,
                        "n_genes": 64, "cells_per_replicate": config["cells_per_replicate"],
                        "target_ids": list(network.regulators), "q1_split": split,
                        "dose": config["dose"], "sampling_times": config["sampling_times"],
                        "conditions": {}, "stationarity_diagnostics": {}, "private_files": {},
                        "warning": "Development generator/scorer vault; expose only allowlisted RNA to learners"}
            _atomic_json(path, manifest)
        private = output / "private/generator.json"
        _atomic_json(private, provenance)
        manifest["private_files"]["private/generator.json"] = _digest(private)
        for name in GENERATOR_SOURCES:
            relative = f"private/source_snapshot/{name}"
            _atomic(output / relative, lambda stream, name=name:
                    stream.write(Path(__file__).with_name(name).read_bytes()))
            manifest["private_files"][relative] = _digest(output / relative)
        _atomic_json(path, manifest)
        try:
            targets = (None, *network.regulators)
            missing = [q for q in targets if str(q) not in manifest["stationarity_diagnostics"]]
            results = _run_jobs(_diagnose, [(params, config, q) for q in missing], workers)
            for target, result in zip(missing, results, strict=True):
                manifest["stationarity_diagnostics"][str(target)] = result.to_dict()
                _atomic_json(path, manifest)
                print(json.dumps({"diagnostic": target, "passed": result.passed,
                                  "burnin": result.burnin}), flush=True)
            diagnostics = manifest["stationarity_diagnostics"]
            manifest["baseline_stationarity"] = diagnostics["None"]
            failures = [q for q in targets if not diagnostics[str(q)]["passed"]]
            if failures:
                manifest.update(status="stationarity_failed", stationarity_failures=failures)
                _atomic_json(path, manifest)
                return manifest
            long_time = max(float(diagnostics[str(q)]["burnin"]) for q in network.regulators)
            if long_time <= 8:
                raise ValueError("Long endpoint must follow the transient observation times")
            manifest.update(common_longterm_time=long_time, status="generating")
            times = (*config["sampling_times"], long_time)
            conditions = (COMMON_BASELINE, *(Condition(q, config["dose"], t)
                                             for q in network.regulators for t in times))
            manifest["planned_timecourse_keys"] = len(conditions)
            from .scmultisim_stationary import StationarityResult
            baseline = StationarityResult(**manifest["baseline_stationarity"])
            jobs = [(output, params, config, baseline, key,
                     diagnostics[str(key.target)] if key.time == long_time else None)
                    for key in conditions if key_id(key) not in manifest["conditions"]]
            _atomic_json(path, manifest)
            for identifier, entry in _run_jobs(_sample, jobs, workers):
                manifest["conditions"][identifier] = entry
                _atomic_json(path, manifest)
                print(json.dumps({"completed": len(manifest["conditions"]), "total": len(conditions),
                                  "condition": entry["condition"]}), flush=True)
            manifest.update(status="complete", stationarity_failures=[],
                            seconds_this_invocation=time.monotonic() - started,
                            n_edges=topology["n_edges"], independent_networks=1,
                            total_cell_observations=3 * config["cells_per_replicate"] * (1 + 2 * (len(conditions) - 1)))
            _atomic_json(path, manifest)
            return manifest
        except Exception as error:
            manifest.update(status="failed", error=repr(error))
            _atomic_json(path, manifest)
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    manifest = generate(json.loads(args.config.read_text()), args.output, workers=args.workers)
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()),
                      "conditions": len(manifest["conditions"])}), flush=True)
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
