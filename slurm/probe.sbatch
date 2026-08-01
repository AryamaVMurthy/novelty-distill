#!/bin/bash
#SBATCH -A priyesh.shukla
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=nd-probe
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

module load u22/cuda/12.4

printf 'job_id=%s\n' "$SLURM_JOB_ID"
printf 'host=%s\n' "$(hostname)"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
printf 'scratch_path=%s\n' "/scratch/$USER"

nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
nvidia-smi topo -m
nvcc --version
df -h "/scratch/$USER"

printf 'end_time=%s\n' "$(date --iso-8601=seconds)"
