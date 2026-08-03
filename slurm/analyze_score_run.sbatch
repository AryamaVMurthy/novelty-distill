#!/bin/bash
# Render strict descriptive diagnostics for a completed score corpus.

#SBATCH --job-name=nd-analyze-scores
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
score_id="${SCORE_ID:-teacher-1k-v1}"
score_namespace="${SCORE_NAMESPACE:-teacher-scores}"
expected_prompts="${EXPECTED_PROMPTS:-1000}"
samples_per_prompt="${SAMPLES_PER_PROMPT:-8}"
for value in "${score_id}" "${score_namespace}"; do
  if [[ -z "${value}" || "${value}" == */* || "${value}" == *..* ]]; then
    echo "score ID and namespace must be path-safe" >&2
    exit 2
  fi
done
for value in "${expected_prompts}" "${samples_per_prompt}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "expected prompts and samples per prompt must be positive integers" >&2
    exit 2
  fi
done

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"
export UV_CACHE_DIR="${scratch_root}/cache/uv-analysis"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

output_root="${scratch_root}/evaluations/score-diagnostics-${score_id}"
mkdir -p "${output_root}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_score_run.py" \
  --score-dir "${scratch_root}/${score_namespace}/${score_id}" \
  --samples-per-prompt "${samples_per_prompt}" \
  --expected-prompts "${expected_prompts}" \
  --output "${output_root}/analysis.json" \
  --markdown "${output_root}/findings.md"
test -s "${output_root}/analysis.json"
test -s "${output_root}/findings.md"
