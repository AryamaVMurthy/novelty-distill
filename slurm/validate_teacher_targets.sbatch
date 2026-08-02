#!/bin/bash
# Validate the frozen production target artifact before releasing training jobs.

#SBATCH --job-name=nd-validate-targets
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
generation_id="${GENERATION_ID:-teacher-1k-v1}"
target_name="${TARGET_NAME:-teacher-targets-tomato1k-v1.json}"
prompts_name="${PROMPTS_NAME:-tomato-open-train-1000.jsonl}"
expected_prompts="${EXPECTED_PROMPTS:-1000}"
for value in "${generation_id}" "${target_name}" "${prompts_name}"; do
  if [[ -z "${value}" || "${value}" == */* || "${value}" == *..* ]]; then
    echo "generation, target, and prompt names must be path-safe" >&2
    exit 2
  fi
done
if ! [[ "${expected_prompts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_PROMPTS must be a positive integer" >&2
  exit 2
fi

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"
export UV_CACHE_DIR="${scratch_root}/cache/uv-analysis"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
flock -u 9
mkdir -p "${scratch_root}/locks"
exec 8>"${scratch_root}/locks/venv-data.lock"
flock -x 8
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

output="${scratch_root}/evaluations/teacher-target-validation-${generation_id}.json"
"${venv_dir}/bin/python" "${repo_dir}/scripts/validate_teacher_targets.py" \
  --targets "${scratch_root}/data/${target_name}" \
  --clustered "${scratch_root}/data/teacher-generations-${generation_id}.jsonl" \
  --prompts "${scratch_root}/data/${prompts_name}" \
  --expected-prompts "${expected_prompts}" \
  --seed 17 \
  --output "${output}"
test -s "${output}"

analysis_root="${scratch_root}/evaluations/teacher-targets-${generation_id}"
mkdir -p "${analysis_root}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_teacher_targets.py" \
  --score-dir "${scratch_root}/teacher-scores/${generation_id}" \
  --clustered "${scratch_root}/data/teacher-generations-${generation_id}.jsonl" \
  --cluster-metadata \
    "${scratch_root}/data/teacher-generations-${generation_id}.jsonl.metadata.json" \
  --targets "${scratch_root}/data/${target_name}" \
  --expected-prompts "${expected_prompts}" \
  --seed 17 \
  --output "${analysis_root}/analysis.json" \
  --markdown "${analysis_root}/findings.md"
test -s "${analysis_root}/analysis.json"
test -s "${analysis_root}/findings.md"
