#!/usr/bin/env bash
# Offline single-GPU training. The optional CPU smoke never reads held RNA.
set -Eeuo pipefail

mode=train
if [[ $# -eq 1 && $1 == --cpu-smoke ]]; then
    mode=cpu-smoke
elif [[ $# -ne 0 ]]; then
    echo "Usage: run_q1_h200.sh [--cpu-smoke]" >&2
    exit 64
fi

dataset=${Q1_DATASET:-/input/q1-modular-64}
output_root=${Q1_OUTPUT_ROOT:-/output}
run_name=${Q1_RUN_NAME:-q1-modular-64-h200-seed20261022}
if [[ $mode == cpu-smoke ]]; then
    run_name=${Q1_RUN_NAME:-q1-modular-64-cpu-smoke-seed20261022}
fi
if [[ ! $run_name =~ ^q1-modular-64-[A-Za-z0-9._-]+$ ]]; then
    echo "Q1_RUN_NAME must be a single q1-modular-64-* directory name" >&2
    exit 64
fi
if [[ ! -f $dataset/manifest.json ]]; then
    echo "Missing public dataset: $dataset/manifest.json" >&2
    exit 66
fi
if [[ ! -d $output_root || ! -w $output_root ]]; then
    echo "Output root must already exist and be writable; mount persistent storage at $output_root" >&2
    exit 73
fi
run_root=$output_root/$run_name
# mkdir must fail on any existing run; never append into or overwrite old runs.
mkdir "$run_root"
(
trap 'status=$?; printf "Task exit status: %s\n" "$status"; exit "$status"' EXIT

python - "$mode" "$run_root/runtime.json" <<'PY'
import json
import platform
from pathlib import Path
import sys
import numpy as np
import torch

mode, destination = sys.argv[1:]
runtime = {"mode": mode, "machine": platform.machine(), "python": platform.python_version(),
           "numpy": np.__version__, "torch": torch.__version__, "cuda_build": torch.version.cuda,
           "cuda_available": torch.cuda.is_available(),
           "note": "Runtime evidence from this host; not a prior H200 verification claim"}
try:
    if mode == "train":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable; refusing an accidental CPU long run")
        # A standard device check, not a separate scientific validation run.
        probe = torch.ones((2, 2), device="cuda", requires_grad=True)
        (probe @ probe).sum().backward()
        torch.cuda.synchronize()
        runtime.update(gpu=torch.cuda.get_device_name(0),
                       capability=list(torch.cuda.get_device_capability(0)),
                       visible_gpu_count=torch.cuda.device_count())
except Exception as error:
    runtime["gpu_probe_error"] = repr(error)
    raise
finally:
    Path(destination).write_text(json.dumps(runtime, indent=2, allow_nan=False) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)
PY

if [[ $mode == cpu-smoke ]]; then
    python /app/tools/scmultisim/q1_h200_smoke.py \
        --dataset "$dataset" --output "$run_root/smoke.json"
else
    python -m grn_experiment.q1_pilot \
        --dataset "$dataset" --output "$run_root/results" \
        --updates 4800 --seed 20261022 --device cuda --threads 4
fi
printf 'Return only this directory (including run.log): %s\n' "$run_root"
) 2>&1 | tee "$run_root/run.log"
