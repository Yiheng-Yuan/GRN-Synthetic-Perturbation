#!/usr/bin/env bash
# Native WSL single-GPU entry: probe first; never silently fall back to CPU.
set -Eeuo pipefail

mode=${1:-probe}
if [[ $# -gt 1 || ! $mode =~ ^(probe|train|resume)$ ]]; then
    echo "Usage: bash tools/scmultisim/run_q1_wsl.sh [probe|train|resume]" >&2
    exit 64
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
dataset=${Q1_DATASET:-$project_root/data/q1-modular-64}
output_root=${Q1_OUTPUT_ROOT:-$project_root/results}
run_name=${Q1_RUN_NAME:-q1-modular-64-wsl-seed20261022}
if [[ ! $run_name =~ ^q1-modular-64-[A-Za-z0-9._-]+$ ]]; then
    echo "Q1_RUN_NAME must be a single q1-modular-64-* directory name" >&2
    exit 64
fi
if [[ ! -f $dataset/manifest.json ]]; then
    echo "Missing public dataset: $dataset/manifest.json" >&2
    exit 66
fi
export PYTHONPATH="$project_root/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
run_root=$output_root/$run_name
mkdir -p "$run_root"
(
trap 'status=$?; printf "Task exit status: %s\n" "$status"; exit "$status"' EXIT
if [[ $mode == probe ]]; then
    python "$project_root/tools/scmultisim/q1_gpu_probe.py" \
        --dataset "$dataset" --output "$run_root/gpu_probe.json"
else
    # Bind the GPU acceptance to this exact public dataset and learner, not
    # merely to an unrelated 2x2 CUDA check or a different machine's report.
    python - "$run_root/gpu_probe.json" "$dataset" <<'PY'
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch
import grn_experiment.q1_pilot as learner

path, dataset = Path(sys.argv[1]), Path(sys.argv[2])
if not path.is_file():
    raise SystemExit("Run the probe successfully before starting training")
report = json.loads(path.read_text())
if report.get("status") != "passed" or not torch.cuda.is_available():
    raise SystemExit("A successful probe and a currently available CUDA GPU are required")
if any(report.get(name) != value for name, value in {
        "condition_microbatch": 1, "effective_conditions": 4, "cells_per_condition": 64}.items()):
    raise SystemExit("GPU probe batch settings differ from the training entry")
if report.get("manifest_sha256") != hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest():
    raise SystemExit("Dataset differs from the GPU probe; use a new run and probe it again")
probe_source = Path(learner.__file__).resolve().parents[2] / "tools/scmultisim/q1_gpu_probe.py"
if report.get("probe_source_sha256") != hashlib.sha256(probe_source.read_bytes()).hexdigest():
    raise SystemExit("GPU probe implementation changed; use a new run and probe again")
expected = ("q1_pilot.py", "q1_pilot_data.py", "protocol.py")
hashes = report.get("source_sha256", {})
for name in expected:
    source = Path(learner.__file__).with_name(name)
    if hashes.get(f"src/grn_experiment/{name}") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise SystemExit(f"Learner differs from the GPU probe: {name}")
runtime = {"python_version": platform.python_version(), "numpy_version": np.__version__,
           "torch_version": str(torch.__version__), "device_name": torch.cuda.get_device_name(0),
           "torch_cuda_version": torch.version.cuda,
           "cuda_capability": list(torch.cuda.get_device_capability(0))}
for name, value in runtime.items():
    if report.get(name) != value:
        raise SystemExit(f"Runtime differs from the GPU probe: {name}; probe a new run")
print("Same-dataset, same-learner GPU probe accepted; full-run memory and convergence are not guaranteed.", flush=True)
PY
    arguments=(--dataset "$dataset" --output "$run_root/results"
               --updates 4800 --seed 20261022 --device cuda --threads 4
               --condition-microbatch 1 --checkpoint-every 60)
    if [[ $mode == resume ]]; then
        arguments+=(--resume)
    fi
    python -m grn_experiment.q1_pilot "${arguments[@]}"
fi
printf 'Result directory: %s\n' "$run_root"
) 2>&1 | tee -a "$run_root/run.log"
