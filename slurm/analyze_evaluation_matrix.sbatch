#!/bin/bash
# Collect all prompt metrics and execute the frozen paired contrast family.

#SBATCH --job-name=nd-analyze-matrix
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"
evaluation_root="${scratch_root}/evaluations/tomato"
output_dir="${evaluation_root}/analysis-seed17"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
flock -u 9

export UV_CACHE_DIR="${scratch_root}/cache/uv-analysis"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
mkdir -p "${output_dir}"

baseline_ids=(
  B1 B2a B2b B2c B3 B4
  C1-human C1-best1 C1-diverse4 C2-human C2-best1 C2-diverse4 C3
  D1 D2 D3 E2 E3 E4
)
collector_inputs=(
  --input "A0=${evaluation_root}/A0-temporal-k16-seed17000.json"
  --input "A1=${evaluation_root}/A1-temporal-k16-seed17000.json"
  --input "A3=${evaluation_root}/A3-temporal-k1.json"
)
for baseline_id in "${baseline_ids[@]}"; do
  eval_id="${baseline_id}-tomato1k-seed17-temporal-k16"
  test -s "${evaluation_root}/${eval_id}.json"
  collector_inputs+=(--input "${baseline_id}=${evaluation_root}/${eval_id}.json")
done
test -s "${evaluation_root}/A0-temporal-k16-seed17000.json"
test -s "${evaluation_root}/A1-temporal-k16-seed17000.json"
test -s "${evaluation_root}/A3-temporal-k1.json"

matrix_path="${output_dir}/prompt-metrics.jsonl"
threshold_matrix_path="${output_dir}/threshold-prompt-metrics.jsonl"
"${venv_dir}/bin/python" "${repo_dir}/scripts/collect_evaluation_metrics.py" \
  "${collector_inputs[@]}" \
  --output "${matrix_path}" \
  --threshold-output "${threshold_matrix_path}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_contrast_matrix.py" \
  --input "${matrix_path}" \
  --threshold-input "${threshold_matrix_path}" \
  --config "${repo_dir}/configs/evaluation/primary_contrasts.yaml" \
  --output "${output_dir}/primary-contrasts.json"
test -s "${output_dir}/prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/threshold-prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/primary-contrasts.json"
