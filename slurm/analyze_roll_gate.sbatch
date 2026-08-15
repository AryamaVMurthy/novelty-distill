#!/bin/bash
#SBATCH --job-name=nd-roll-analysis
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

repo_ref="${REPO_REF:-codex/semantic-seed-runs}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9

output="${scratch_root}/evaluations/semantic-seed/gate1-roll-analysis.json"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_roll_gate.py" \
  --ordinary "${scratch_root}/data/teacher-generations-a0-ordinary-v1.jsonl" \
  --seeded "${scratch_root}/data/teacher-generations-a0-seeded-v1.jsonl" \
  --input-seed-repeats 2 \
  --output "${output}"
test -s "${output}"
