#!/usr/bin/env bash
# Example only: partition/account/container integration are site-specific.
#SBATCH --job-name=q1-modular64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-q1-%j.log
set -Eeuo pipefail

# Set these absolute host directories when submitting, before sbatch.
: "${Q1_HOST_DATASET:?Directory containing the public manifest.json and observations/q1}"
: "${Q1_HOST_OUTPUT:?Persistent writable host output directory}"
image=${Q1_IMAGE:-grn-q1-modular:20261023-h200-dev1}
# Use the allocated host GPU index/UUID, not an assumed device or every host GPU.
gpu_device=${Q1_GPU_DEVICE:-${SLURM_JOB_GPUS:-}}
if [[ ! $gpu_device =~ ^[0-9]+$ && ! $gpu_device =~ ^GPU-[A-Fa-f0-9-]+$ ]]; then
    echo "Set Q1_GPU_DEVICE to the single allocated host GPU index or GPU-UUID" >&2
    exit 64
fi
if ! command -v docker >/dev/null; then
    echo "This site does not expose Docker in jobs; use its image submission interface and the documented in-container command." >&2
    exit 69
fi
mkdir -p "$Q1_HOST_OUTPUT"
docker image inspect "$image" >/dev/null
srun --ntasks=1 docker run --rm --pull=never --network none --read-only \
    --gpus "device=$gpu_device" --shm-size=1g --tmpfs /tmp:rw,size=2g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$Q1_HOST_DATASET,dst=/input/q1-modular-64,readonly" \
    --mount "type=bind,src=$Q1_HOST_OUTPUT,dst=/output" \
    -e "Q1_RUN_NAME=q1-modular-64-h200-job${SLURM_JOB_ID}" \
    "$image"
