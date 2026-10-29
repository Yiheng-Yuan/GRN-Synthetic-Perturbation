"""Run the registered RNA-current/history comparison sequentially on one GPU.

The shell wrapper owns the output directory and log. This module binds a run
to its code, public data, settings and runtime, and validates completed arms
before skipping them on resume. It does not synthesize data or tune on scores.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from grn_experiment.q1_pilot import PilotConfig, build_model, _learner_sources
from grn_experiment.q1_pilot_data import load_modular_q1, split_registered_targets
from q1_gpu_probe import run_probe


PROJECT = Path(__file__).resolve().parents[2]
ARMS = (("rna_current", 64), ("rna_history", 51))
MANIFEST_SHA256 = "7ce56f42b9b3ebe3206cb6591dba2536dfdb6511128cfce143a83751c97d50de"


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def arm_config(kind, hidden, times):
    return PilotConfig(updates=4800, conditions_per_update=4, condition_microbatch=1,
        cells=64, projections=24, hidden=hidden, step=.25, learning_rate=.001,
        effect_weight=1., target_weight=0., seed=20261022, fit_times=times,
        model_kind=kind, history_lag=1.)


def registration(dataset):
    """Validate all public RNA hashes, never load private simulator arrays."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an accidental CPU training run")
    if checksum(dataset / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("This comparison requires the unchanged 64-gene dataset from 2026-10-23")
    data = load_modular_q1(dataset)
    fitting, held = split_registered_targets(data)
    times = tuple(sorted({key.time for key in fitting}))
    if times != (.25, 1., 2., 4., 8., 64.):
        raise ValueError("Unexpected sampling grid")
    configs = {kind: arm_config(kind, width, times) for kind, width in ARMS}
    paths = [f"src/grn_experiment/{name}" for name in
             (*_learner_sources(configs["rna_history"]), "__init__.py")]
    paths += ["tools/scmultisim/" + name for name in
              ("q1_rna_run.py", "q1_gpu_probe.py", "run_q1_rna.sh")]
    return canonical({
        "schema": "q1-rna-comparison-v1", "manifest_sha256": MANIFEST_SHA256,
        "input_provenance": data.provenance,
        "sources": {name: checksum(PROJECT / name) for name in paths},
        "configs": {kind: asdict(cfg) for kind, cfg in configs.items()},
        "parameter_counts": {kind: sum(p.numel() for p in build_model(64, cfg).parameters())
                             for kind, cfg in configs.items()},
        "fit_targets": sorted({key.target for key in fitting}),
        "held_targets": sorted({key.target for key in held}),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "machine": platform.machine(),
                    "gpu": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0))},
        "scope": "previously inspected one-network development data; no biological Markov-order claim",
        "history": "own generated RNA at t-1; constant initial RNA at t<=0",
        "loss": "same existing 24-projection population/control plus standardized response loss",
        "threads": 4, "checkpoint_every": 60,
    })


def verify_completed(output, registered, kind):
    """Accept only intact, matching, numerically checked complete arm results."""
    report = json.loads((output / "report.json").read_text())
    plan = json.loads((output / "run_plan.json").read_text())
    hashes = json.loads((output / "SHA256SUMS.json").read_text())
    required = {"model.pt", "training_checkpoint.pt", "report.json", "run_plan.json",
                "history.json", "scores.json", "predictions.npz", "progress.jsonl"}
    if not required <= set(hashes):
        raise ValueError(f"{kind}: incomplete result checksum manifest")
    for name, digest in hashes.items():
        if Path(name).name != name or checksum(output / name) != digest:
            raise ValueError(f"{kind}: result checksum mismatch: {name}")
    if (report.get("status") != "completed" or report.get("integration_check", {}).get("passed") is not True
            or plan.get("config") != registered["configs"][kind]
            or report.get("config") != plan["config"]
            or report.get("model_kind") != kind or plan.get("model_kind") != kind
            or plan.get("input_provenance") != registered["input_provenance"]
            or plan.get("fit_targets") != registered["fit_targets"]
            or plan.get("test_targets") != registered["held_targets"]):
        raise ValueError(f"{kind}: completed result disagrees with registration or numerical check")
    for name in _learner_sources(PilotConfig(model_kind=kind)):
        expected = registered["sources"]["src/grn_experiment/" + name]
        if plan.get("source_sha256", {}).get(name) != expected:
            raise ValueError(f"{kind}: registered learner source differs")
    for name, digest in plan["source_snapshot"]["files"].items():
        if Path(name).name != name or checksum(output / "source_snapshot/grn_experiment" / name) != digest:
            raise ValueError(f"{kind}: archived learner source differs")
    return report


def probe_arm(dataset, output, registered, kind, hidden):
    """Measure a fresh longest-time update using this exact arm and discard it."""
    path = output / f"probe-{kind}-{time.time_ns()}.json"
    report = run_probe(dataset, path, model_kind=kind, hidden=hidden, history_lag=1.)
    print(json.dumps(report, indent=2), flush=True)
    expected = {**registered["configs"][kind], "updates": 1, "fit_times": [64.]}
    if report.get("status") != "passed" or report.get("config") != expected:
        raise RuntimeError(f"{kind}: GPU acceptance failed; see {path.name}")
    if report["manifest_sha256"] != registered["manifest_sha256"]:
        raise RuntimeError("GPU probe dataset differs")
    for name, digest in report["source_sha256"].items():
        if registered["sources"].get(name) != digest:
            raise RuntimeError("GPU probe learner differs")
    # Probe allocations are disposable; release the cache before real fitting.
    torch.cuda.empty_cache()


def train_arm(dataset, output, kind, hidden, resume):
    argv = [sys.executable, "-m", "grn_experiment.q1_pilot",
            "--dataset", str(dataset), "--output", str(output),
            "--model-kind", kind, "--hidden", str(hidden), "--history-lag", "1",
            "--updates", "4800", "--seed", "20261022", "--device", "cuda",
            "--threads", "4", "--condition-microbatch", "1", "--checkpoint-every", "60"]
    if output.exists():
        if not resume or not (output / "training_checkpoint.pt").is_file():
            raise ValueError(f"{kind}: existing unfinished output has no resumable checkpoint; preserve it and use a new output root")
        argv.append("--resume")
    subprocess.run(argv, check=True)


def run(args):
    registered = registration(args.dataset)
    path = args.output / "comparison_plan.json"
    if args.mode == "resume":
        if not path.is_file() or json.loads(path.read_text()) != registered:
            raise ValueError("Comparison registration changed: data, source, configuration or GPU/runtime")
    else:
        if path.exists():
            raise ValueError("Comparison already registered; use resume")
        write_json(path, registered)
    reports = {}
    for kind, width in ARMS:
        destination = args.output / kind
        if (destination / "report.json").is_file():
            reports[kind] = verify_completed(destination, registered, kind)
            print(f"Verified completed arm: {kind}; no repeat training", flush=True)
            continue
        probe_arm(args.dataset, args.output, registered, kind, width)
        if args.mode == "probe":
            continue
        train_arm(args.dataset, destination, kind, width, args.mode == "resume")
        reports[kind] = verify_completed(destination, registered, kind)
    if args.mode != "probe":
        write_json(args.output / "comparison.json", {
            "status": "completed", "scope": registered["scope"],
            "parameter_counts": registered["parameter_counts"],
            "metrics": {kind: report["metrics"] for kind, report in reports.items()},
            "seconds": {kind: report["seconds"] for kind, report in reports.items()},
            "timing_note": "each arm's latest process; prior interrupted attempts are excluded",
            "conclusion": "training and numerical checks completed; scientific benefit requires metric review"})
    print(f"Return this entire result directory only: {args.output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "resume", "probe"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    try:
        run(args)
    except Exception as error:
        write_json(args.output / f"failure-{time.time_ns()}.json",
                   {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
