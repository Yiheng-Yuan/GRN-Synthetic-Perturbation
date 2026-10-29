"""Measure one longest-time GPU training update, not a complete training run.

Only the common baseline and the first four preregistered training targets
are read. Fitting uses repeats 0/1, never held targets or repeat 2. No model
or scientific prediction is saved; a failed probe still leaves its report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch

from grn_experiment.protocol import COMMON_BASELINE, Condition
from grn_experiment.q1_pilot import PilotConfig, fit_pilot
from grn_experiment.q1_pilot_data import (
    _MAX_MANIFEST_BYTES, _MAX_OBSERVATION_BYTES, _decode, _diagnostic,
    _directory_payload, _json, _modular_layout, _record, _select_entry,
)


PROJECT = Path(__file__).resolve().parents[2]
LEARNER_SOURCES = (
    "src/grn_experiment/q1_pilot.py",
    "src/grn_experiment/q1_pilot_data.py",
    "src/grn_experiment/protocol.py",
)


def load_probe_data(dataset: Path) -> tuple[np.ndarray, dict, dict]:
    """Hash/decode exactly five public RNA files, with no broad grid loader."""
    dataset = Path(dataset).resolve()
    payload = _directory_payload(dataset, "manifest.json", _MAX_MANIFEST_BYTES)
    manifest = _json(payload)
    _, shape, conditions, registration = _modular_layout(manifest)
    _, baseline_diagnostic = _diagnostic(manifest.get("baseline_stationarity"), None, 0.)
    duration = manifest.get("common_longterm_time")
    if type(duration) not in (int, float) or not np.isfinite(duration) or duration <= 8:
        raise ValueError("A finite actual common long-term sampling time greater than 8 is required")
    targets = registration["train"][:4]
    selected = [COMMON_BASELINE, *(Condition(target, .7, float(duration)) for target in targets)]
    snapshots, verified = {}, {}
    for condition in selected:
        relative, checksum, entry = _select_entry(conditions, _record(condition), "observations/q1")
        if condition != COMMON_BASELINE:
            burnin, diagnostic = _diagnostic(entry.get("diagnostic"), condition.target, .7)
            actual = entry.get("actual_sample_time")
            if (type(actual) not in (int, float) or actual != duration or burnin > duration
                    or diagnostic["tau"] != baseline_diagnostic["tau"]
                    or diagnostic["parameter_sha256"] != baseline_diagnostic["parameter_sha256"]):
                raise ValueError("Long-term sample time or passed diagnostic is inconsistent")
        observation = _directory_payload(dataset, relative, _MAX_OBSERVATION_BYTES)
        if hashlib.sha256(observation).hexdigest() != checksum:
            raise ValueError(f"Observation checksum mismatch: {relative}")
        snapshots[condition] = _decode(observation, condition, shape)
        verified[relative] = checksum
    # The NPZ format stores three repeats, but only the first two enter fitting.
    baseline = snapshots.pop(COMMON_BASELINE).treated[:2].reshape(-1, shape[-1])
    provenance = {
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "verified_observation_sha256": verified,
        "verified_observation_count": len(verified),
        "n_genes": shape[-1], "actual_time": float(duration), "target_ids": targets,
        "fit_replicates": [0, 1], "held_targets_read": False,
        "third_replicate_used_for_fitting": False, "private_files_read": False,
    }
    return baseline, snapshots, provenance


def _gpu_memory(report: dict, device: torch.device) -> None:
    """Preserve peaks and remaining headroom on success and ordinary OOM."""
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    report["free_bytes"] = free
    report["total_bytes"] = total


def run_probe(dataset: Path, output: Path, *, model_kind="latent_neural_ode",
              hidden=64, history_lag=1.0) -> dict:
    """Reserve a fresh report, require CUDA, and measure one optimizer update.

    The reported elapsed time includes model/optimizer setup and the complete
    forward/backward/update. One successful update is not a peak-memory or
    stability guarantee for all subsequent updates of a complete fit.
    """
    # Exclusive creation happens before work. Never overwrite a prior report.
    sources = LEARNER_SOURCES + (("src/grn_experiment/q1_rna.py",)
                                 if model_kind != "latent_neural_ode" else ())
    with Path(output).open("x", encoding="utf-8") as stream:
        report = {
            "schema": "q1-gpu-probe-v1", "status": "running",
            "scope": "one longest-time forward/backward optimizer update; not full training or scientific validation",
            "dataset": str(Path(dataset).resolve()),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "source_sha256": {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                              for name in sources},
            "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "condition_microbatch": 1, "effective_conditions": 4, "cells_per_condition": 64,
            "elapsed_seconds": None,
        }
        start, device = None, None
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; refusing an accidental CPU longest-time run")
            device = torch.device("cuda:0")
            report["device_name"] = torch.cuda.get_device_name(device)
            report["cuda_capability"] = list(torch.cuda.get_device_capability(device))
            baseline, observations, provenance = load_probe_data(dataset)
            report.update(provenance)
            config = PilotConfig(updates=1, conditions_per_update=4, condition_microbatch=1,
                                 cells=64, projections=24, hidden=hidden, step=.25,
                                 learning_rate=.001, effect_weight=1., target_weight=0.,
                                 seed=20261022, fit_times=(provenance["actual_time"],),
                                 model_kind=model_kind, history_lag=history_lag)
            report["config"] = {**asdict(config), "fit_times": list(config.fit_times)}
            torch.set_num_threads(4)
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            report.update(free_before_bytes=free, total_before_bytes=total)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            _, history = fit_pilot(baseline, observations, config=config, device=str(device))
            torch.cuda.synchronize(device)
            report["elapsed_seconds"] = time.perf_counter() - start
            _gpu_memory(report, device)
            if (len(history) != 1 or history[0].get("time") != provenance["actual_time"]
                    or not np.isfinite(history[0].get("loss", float("nan")))):
                raise RuntimeError("The probe did not return one finite longest-time update")
            report.update(status="passed", finite_loss=True, update=history[0])
        except (Exception, KeyboardInterrupt) as error:
            if start is not None and report["elapsed_seconds"] is None:
                report["elapsed_seconds"] = time.perf_counter() - start
            report.update(status="failed", error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
            if device is not None:
                try:
                    _gpu_memory(report, device)
                except Exception as memory_error:
                    report["memory_reporting_error"] = str(memory_error)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="A new JSON file in an existing writable directory")
    parser.add_argument("--model-kind", choices=("latent_neural_ode", "rna_current", "rna_history"),
                        default="latent_neural_ode")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--history-lag", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = run_probe(args.dataset, args.output, model_kind=args.model_kind,
                       hidden=args.hidden, history_lag=args.history_lag)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
