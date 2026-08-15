#!/bin/bash
#SBATCH --job-name=nd-seed-cluster
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

run_ids=(SS0-C1 SS0D-DIVERSE SS1-CR SS2-GSC SS3-TL SS4-GSC-TL)
index="${SLURM_ARRAY_TASK_ID:?semantic seed clustering requires an array task}"
if ((index < 0 || index >= ${#run_ids[@]})); then
  echo "SLURM_ARRAY_TASK_ID is outside the semantic seed matrix" >&2
  exit 2
fi
run_id="${run_ids[$index]}"
export GENERATION_JOB_ID="${run_id}-short-v1"
export TARGET_NAME="${run_id}-short-diagnostic-targets.json"
export SCORE_NAMESPACE="semantic-seed-short-scores"
export SAMPLES_PER_PROMPT=8
exec bash "/scratch/${USER}/novelty-distill/repo/slurm/cluster_teacher.sbatch"
