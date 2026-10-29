#!/usr/bin/env bash
# Two RNA-state arms, one GPU, separate checkpoints; train is the default.
set -Eeuo pipefail
mode=${1:-train}
if [[ $# -gt 1 || ! $mode =~ ^(train|resume|probe)$ ]]; then
    echo 'Usage: bash tools/scmultisim/run_q1_rna.sh [train|resume|probe]' >&2
    exit 64
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
dataset=${Q1_DATASET:-/data/q1-modular-64}
output_root=${Q1_OUTPUT_ROOT:-/output}
if [[ ! -f $dataset/manifest.json ]]; then
    echo "Missing public dataset: $dataset/manifest.json" >&2
    exit 66
fi
if [[ ! -d $output_root ]]; then
    echo 'Q1_OUTPUT_ROOT must be an existing persistent output directory' >&2
    exit 73
fi
run_root=$output_root/q1-rna-comparison-seed20261022
if [[ $mode == resume ]]; then
    if [[ ! -f $run_root/comparison_plan.json ]]; then
        echo 'No registered comparison to resume' >&2
        exit 66
    fi
else
    # No -p: protect completed and interrupted comparisons from replacement.
    mkdir "$run_root"
fi
export PYTHONPATH="$project_root/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
(
trap 'status=$?; printf "Task exit status: %s\n" "$status"; exit "$status"' EXIT
python "$project_root/tools/scmultisim/q1_rna_run.py" "$mode" \
    --dataset "$dataset" --output "$run_root"
) 2>&1 | tee -a "$run_root/run.log"
