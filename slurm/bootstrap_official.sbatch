#!/bin/bash
#SBATCH --job-name=nd-official
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"
requested="${OFFICIAL_REPOS:-trl opsd gem minillm distillm}"

"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"

read -r -a names <<<"${requested}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/bootstrap_official.py" \
  --manifest "${repo_dir}/third_party/manifest.yaml" \
  --destination "${scratch_root}/official" \
  "${names[@]}"
