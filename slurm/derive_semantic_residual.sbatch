#!/bin/bash
#SBATCH --job-name=nd-residual
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/semantic-seed-runs}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
teacher_clustered="${scratch_root}/data/teacher-generations-teacher-semantic-seed-128-v1.jsonl"
student_clustered="${scratch_root}/data/teacher-generations-a0-student-coverage-128-v1.jsonl"
teacher_bank="${scratch_root}/data/semantic-seed-teacher-bank-128.jsonl"
student_bank="${scratch_root}/data/semantic-seed-student-bank-128.jsonl"
output="${scratch_root}/data/teacher-targets-semantic-residual-k16-g1.json"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9

"${venv_dir}/bin/python" "${repo_dir}/scripts/build_semantic_bank.py" \
  --clustered "${teacher_clustered}" \
  --output "${teacher_bank}" \
  --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml" \
  --embedding-cache-dir "${scratch_root}/cache/embeddings" \
  --role teacher
"${venv_dir}/bin/python" "${repo_dir}/scripts/build_semantic_bank.py" \
  --clustered "${student_clustered}" \
  --output "${student_bank}" \
  --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml" \
  --embedding-cache-dir "${scratch_root}/cache/embeddings" \
  --role student
"${venv_dir}/bin/python" "${repo_dir}/scripts/derive_coverage_residual_targets.py" \
  --teacher-bank "${teacher_bank}" \
  --student-bank "${student_bank}" \
  --output "${output}" \
  --kappa 16 \
  --gamma 1 \
  --epsilon 0.001 \
  --minimum-quality 0.75
test -s "${output}"
