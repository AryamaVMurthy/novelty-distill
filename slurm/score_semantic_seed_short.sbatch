#!/bin/bash
#SBATCH --job-name=nd-seed-score
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

run_ids=(SS0-C1 SS0D-DIVERSE SS1-CR SS2-GSC SS3-TL SS4-GSC-TL)
generation_configs=(
  configs/generation/semantic_seed_short_lora.yaml
  configs/generation/semantic_seed_short_lora.yaml
  configs/generation/semantic_seed_short_lora.yaml
  configs/generation/semantic_seed_short_seeded_lora.yaml
  configs/generation/semantic_seed_short_lora.yaml
  configs/generation/semantic_seed_short_seeded_lora.yaml
)
index="${SLURM_ARRAY_TASK_ID:?semantic seed scoring requires an array task}"
if ((index < 0 || index >= ${#run_ids[@]})); then
  echo "SLURM_ARRAY_TASK_ID is outside the semantic seed matrix" >&2
  exit 2
fi
run_id="${run_ids[$index]}"
export GENERATION_JOB_ID="${run_id}-short-v1"
export GENERATION_NAME="semantic-seed-short-eval"
export GENERATION_CONFIG="${generation_configs[$index]}"
export PROMPTS_NAME="semantic-seed-eval-32.jsonl"
export SCORE_NAMESPACE="semantic-seed-short-scores"
export SCORE_CONCURRENCY=8
exec bash "/scratch/${USER}/novelty-distill/repo/slurm/score_teacher.sbatch"
