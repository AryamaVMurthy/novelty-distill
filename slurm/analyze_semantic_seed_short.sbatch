#!/bin/bash
#SBATCH --job-name=nd-seed-select
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_semantic_seed_matrix.py" \
  --cluster-root "${scratch_root}/data" \
  --output "${scratch_root}/evaluations/semantic-seed/short-matrix-selection.json" \
  --validity-tolerance 0.05
