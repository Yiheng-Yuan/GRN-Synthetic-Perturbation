"""Run a fixed four-weight explicit-GRN development comparison on one GPU.

Only public manifest metadata is read here. The trainer owns RNA allowlisting,
the internal target split, checkpoints and prediction scores. The selected arm
is then refitted on all 32 original training targets before held-target scoring.
This entry point never reads truth or selects using held-target responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[2]
SEED = 20261022
ARMS = (("sparse-0", 0.0), ("sparse-1e-5", 1e-5),
        ("sparse-1e-4", 1e-4), ("sparse-1e-3", 1e-3))
MANIFEST_SHA256 = "7ce56f42b9b3ebe3206cb6591dba2536dfdb6511128cfce143a83751c97d50de"
SOURCE_PATHS = tuple(f"src/grn_experiment/{name}" for name in
                     ("q1_graph.py", "q1_graph_train.py", "q1_pilot_data.py",
                      "q1_pilot_evaluation.py", "protocol.py", "__init__.py")) + (
    "tools/scmultisim/q1_graph_run.py", "tools/scmultisim/run_q1_graph.sh")


def checksum(path: Path) -> str:
    """Hash a file without retaining a checkpoint-sized bytes object."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def arm_config(weight: float, updates: int) -> dict:
    return {"updates": updates, "cells": 64, "conditions_per_update": 4,
            "condition_microbatch": 1, "hidden": 8, "step": .25,
            "seed": SEED, "lambda_distribution": .1, "lambda_sparse": weight}


def registration(dataset: Path, updates: int) -> dict:
    """Bind the unchanged public dataset, code and runtime before any update."""
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an accidental CPU training run")
    digest = checksum(dataset / "manifest.json")
    if digest != MANIFEST_SHA256:
        raise ValueError("This comparison requires the unchanged 64-gene development dataset")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("n_genes") != 64:
        raise ValueError("Expected 64 public gene coordinates")
    return {
        "schema": "q1-graph-sweep-v1", "manifest_sha256": digest, "n_genes": 64,
        "sources": {name: checksum(PROJECT / name) for name in SOURCE_PATHS},
        "configs": {name: arm_config(weight, updates) for name, weight in ARMS},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "machine": platform.machine(), "gpu": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0))},
        "selection": {"metric": "validation_expression", "absolute_tie_tolerance": 1e-6,
                      "tie_breaks": ["fewest_binary_edges", "smaller_lambda_sparse"],
                      "scope": "internal training-target validation only; no held-target scores"},
        "scope": "one previously inspected development network; not a formal blind experiment",
        "threads": 4, "checkpoint_every": 60,
    }


def trainer_command(dataset: Path, output: Path, config: dict, *,
                    probe: bool = False, resume: bool = False,
                    refit: bool = False) -> list[str]:
    argv = [sys.executable, "-m", "grn_experiment.q1_graph_train",
            "--dataset", str(dataset), "--output", str(output),
            "--device", "cuda", "--threads", "4", "--checkpoint-every", "60"]
    for key, value in config.items():
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    if probe:
        argv.append("--probe")
    if resume:
        argv.append("--resume")
    if refit:
        argv.append("--refit")
    return argv


def verify_completed(output: Path, registered: dict, name: str, *,
                     refit: bool = False) -> dict:
    """Resume may skip an arm only after all recorded artifacts are verified."""
    import numpy as np

    hashes = json.loads((output / "SHA256SUMS.json").read_text())
    required = {"report.json", "run_plan.json", "model.pt", "learned_graph.npz"}
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError(f"{name}: incomplete result checksum manifest")
    for relative, digest in hashes.items():
        path = PurePosixPath(relative)
        if (not relative or "\\" in relative or "\0" in relative or path.is_absolute()
                or ".." in path.parts or str(path) != relative
                or not (output / relative).resolve().is_relative_to(output.resolve())
                or checksum(output / relative) != digest):
            raise ValueError(f"{name}: result checksum mismatch: {relative}")
    plan = json.loads((output / "run_plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    expected = registered["configs"][name]
    expected_sources = {Path(path).name: digest for path, digest in registered["sources"].items()
                        if path.startswith("src/grn_experiment/")}
    runtime_keys = ("python", "numpy", "torch", "machine", "gpu", "cuda", "capability")
    for value in (plan, report):
        if value.get("source_sha256", {}) != expected_sources:
            raise ValueError(f"{name}: completed learner source differs from registration")
        actual_runtime = value.get("runtime", {})
        if any(key not in actual_runtime or actual_runtime[key] != registered["runtime"][key]
               for key in runtime_keys if key in registered.get("runtime", {})):
            raise ValueError(f"{name}: completed runtime differs from registration")
        config = value.get("config", {})
        if any(config.get(key) != setting for key, setting in expected.items()):
            raise ValueError(f"{name}: completed configuration differs from registration")
        provenance = value.get("provenance", value.get("input_provenance", {}))
        if provenance.get("manifest_sha256") != registered["manifest_sha256"]:
            raise ValueError(f"{name}: completed dataset differs from registration")
    score = report.get("validation_expression")
    valid_score = score is None if refit else (
        not isinstance(score, bool) and isinstance(score, (int, float))
        and math.isfinite(score) and score >= 0)
    if (report.get("status") != "completed" or not valid_score
            or any(value.get("refit", False) != refit for value in (plan, report))):
        raise ValueError(f"{name}: incomplete run or invalid validation expression score")
    with np.load(output / "learned_graph.npz", allow_pickle=False) as graph:
        w, h, z = (graph[key] for key in ("W", "H", "Z"))
        shape = (registered["n_genes"],) * 2
        if (any(array.shape != shape or not np.isfinite(array).all() for array in (w, h, z))
                or not np.isin(w, (0, 1)).all() or np.any(np.diag(w))
                or not np.array_equal(z, w * h)):
            raise ValueError(f"{name}: graph artifact violates the W/H/Z contract")
        edges = int(w.sum())
    return {"validation_expression": None if refit else float(score), "binary_edges": edges,
            "lambda_sparse": expected["lambda_sparse"]}


def select_arm(reports: dict[str, dict], registered: dict) -> dict:
    """Apply the registered rule, without consulting any held-target metrics."""
    minimum = min(report["validation_expression"] for report in reports.values())
    tolerance = registered["selection"]["absolute_tie_tolerance"]
    tied = [name for name, report in reports.items()
            if report["validation_expression"] <= minimum + tolerance]
    chosen = min(tied, key=lambda name: (reports[name]["binary_edges"],
                                       reports[name]["lambda_sparse"]))
    return {"selected_arm": chosen, "rule": registered["selection"],
            "candidates": reports,
            "note": "development selection only; chosen settings are refitted on 32 training targets; independent graph scoring remains separate"}


def probe(dataset: Path, output: Path, registered: dict) -> None:
    """One actual longest-time forward/backward update, separate from fitting."""
    config = registered["configs"]["sparse-1e-4"]
    subprocess.run(trainer_command(dataset, output, config, probe=True), check=True)
    report = json.loads((output / "probe.json").read_text())
    if report.get("status") != "passed":
        raise RuntimeError("Graph GPU probe failed; full training was not started")


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    registered = registration(args.dataset, args.updates)
    path = args.output / "comparison_plan.json"
    if args.mode == "resume":
        if not path.is_file() or json.loads(path.read_text()) != registered:
            raise ValueError("Comparison registration changed: data, code, settings or runtime")
    else:
        if path.exists() or any(p.name != "run.log" for p in args.output.iterdir()):
            raise ValueError("Output already contains results; preserve it and use resume or a new output root")
        write_json(path, registered)
    if args.mode == "probe":
        probe(args.dataset, args.output / "probe", registered)
        print(f"GPU update completed; full training was not run. Archive this result directory only: {args.output}", flush=True)
        return
    reports = {}
    pending = []
    for name, _ in ARMS:
        destination = args.output / name
        if (destination / "report.json").is_file():
            reports[name] = verify_completed(destination, registered, name)
            print(f"Verified completed arm: {name}; no repeat training", flush=True)
        else:
            if destination.exists() and (args.mode != "resume" or not
                                         (destination / "training_checkpoint.pt").is_file()):
                raise ValueError(f"{name}: unfinished output has no resumable checkpoint; preserve it and use a new output root")
            pending.append(name)
    if pending:
        probe(args.dataset, args.output / f"probe-{time.time_ns()}", registered)
    for name in pending:
        destination = args.output / name
        subprocess.run(trainer_command(args.dataset, destination, registered["configs"][name],
                                      resume=destination.exists()), check=True)
        reports[name] = verify_completed(destination, registered, name)
    selected = select_arm(reports, registered)
    write_json(args.output / "selection.json", selected)
    refit_name = selected["selected_arm"]
    destination = args.output / "final_refit"
    if (destination / "report.json").is_file():
        final = verify_completed(destination, registered, refit_name, refit=True)
        print("Verified completed final_refit; no repeat training", flush=True)
    else:
        if destination.exists() and (args.mode != "resume" or not
                                     (destination / "training_checkpoint.pt").is_file()):
            raise ValueError("final_refit: unfinished output has no resumable checkpoint; preserve it and use a new output root")
        if not pending:
            probe(args.dataset, args.output / f"probe-{time.time_ns()}", registered)
        subprocess.run(trainer_command(args.dataset, destination,
                                      registered["configs"][refit_name], refit=True,
                                      resume=destination.exists()), check=True)
        final = verify_completed(destination, registered, refit_name, refit=True)
    write_json(args.output / "comparison.json", {
        "status": "completed", "scope": registered["scope"], "arms": reports,
        "selected_arm": refit_name, "final_refit": final,
        "conclusion": "training outputs verified; network recovery requires independent truth-side scoring"})
    print(f"Archive this result directory only: {args.output}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "train", "resume"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=4800)
    args = parser.parse_args(argv)
    if args.updates < 1:
        parser.error("--updates must be positive")
    try:
        run(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output / f"failure-{time.time_ns()}.json", {
                "status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
