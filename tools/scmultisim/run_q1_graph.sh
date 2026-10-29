#!/usr/bin/env bash
# Four fixed weights then the selected refit, sequentially on Ubuntu CUDA.
set -Eeuo pipefail
mode=${1:-train}
if [[ $# -eq 1 && ( $mode == --help || $mode == -h ) ]]; then
    echo 'Usage: bash tools/scmultisim/run_q1_graph.sh [probe|train|resume]'
    echo 'Optional environment: Q1_DATASET, Q1_OUTPUT_ROOT, Q1_UPDATES (default 4800).'
    exit 0
fi
if [[ $# -gt 1 || ! $mode =~ ^(probe|train|resume)$ ]]; then
    echo 'Usage: bash tools/scmultisim/run_q1_graph.sh [probe|train|resume]' >&2
    exit 64
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
dataset=${Q1_DATASET:-/home/research/grn-q1-wsl-20261023/data/q1-modular-64}
output_root=${Q1_OUTPUT_ROOT:-/home/research/grn-q1-graph-results}
updates=${Q1_UPDATES:-4800}
if [[ ! -f $dataset/manifest.json ]]; then
    echo "Missing public dataset: $dataset/manifest.json" >&2
    exit 66
fi
if [[ ! $updates =~ ^[1-9][0-9]*$ ]]; then
    echo 'Q1_UPDATES must be a positive integer' >&2
    exit 64
fi
mkdir -p "$output_root"
run_root=$output_root/q1-graph-sweep-seed20261022
if [[ $mode == probe ]]; then
    run_root=$output_root/q1-graph-probe-seed20261022
fi
if [[ $mode == resume ]]; then
    if [[ ! -f $run_root/comparison_plan.json ]]; then
        echo 'No registered graph comparison to resume' >&2
        exit 66
    fi
else
    # No -p: protect completed and interrupted outputs from replacement.
    mkdir "$run_root"
fi
export PYTHONPATH="$project_root/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
(
    trap 'status=$?; printf "Task exit status: %s\n" "$status"; exit "$status"' EXIT
    python "$project_root/tools/scmultisim/q1_graph_run.py" "$mode" \
        --dataset "$dataset" --output "$run_root" --updates "$updates"
) 2>&1 | tee -a "$run_root/run.log"
