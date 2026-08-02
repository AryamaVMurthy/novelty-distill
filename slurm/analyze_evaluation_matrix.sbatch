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
taste_root="${scratch_root}/research-taste"
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
mkdir -p "${scratch_root}/locks"
exec 8>"${scratch_root}/locks/venv-data.lock"
flock -x 8
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8
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
taste_inputs=(
  --input "A0=${taste_root}/A0-temporal-k16-seed17000"
  --input "A1=${taste_root}/A1-temporal-k16-seed17000"
  --input "A3=${taste_root}/A3-temporal-k1"
)
for baseline_id in "${baseline_ids[@]}"; do
  eval_id="${baseline_id}-tomato1k-seed17-temporal-k16"
  test -s "${evaluation_root}/${eval_id}.json"
  collector_inputs+=(--input "${baseline_id}=${evaluation_root}/${eval_id}.json")
  test -d "${taste_root}/${eval_id}"
  taste_inputs+=(--input "${baseline_id}=${taste_root}/${eval_id}")
done
test -s "${evaluation_root}/A0-temporal-k16-seed17000.json"
test -s "${evaluation_root}/A1-temporal-k16-seed17000.json"
test -s "${evaluation_root}/A3-temporal-k1.json"
test -d "${taste_root}/A0-temporal-k16-seed17000"
test -d "${taste_root}/A1-temporal-k16-seed17000"
test -d "${taste_root}/A3-temporal-k1"

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
  --output "${output_dir}/primary-contrasts.json" \
  --markdown "${output_dir}/findings.md"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_research_taste.py" \
  "${taste_inputs[@]}" \
  --human-method A3 \
  --teacher-method A1 \
  --config "${repo_dir}/configs/evaluation/research_taste.yaml" \
  --output "${output_dir}/research-taste-secondary.json" \
  --markdown "${output_dir}/research-taste-findings.md"
"${venv_dir}/bin/python" "${repo_dir}/scripts/prepare_research_taste_calibration.py" \
  "${taste_inputs[@]}" \
  --prompts "${scratch_root}/data/tomato-open-test-1658.jsonl" \
  --generation-config "${repo_dir}/configs/generation/eval_qwen3_4b.yaml" \
  --size 150 \
  --seed 17 \
  --packet "${output_dir}/research-taste-human-calibration.jsonl" \
  --key "${output_dir}/research-taste-human-calibration-key.json" \
  --instructions "${output_dir}/research-taste-human-calibration-instructions.md"
test -s "${output_dir}/prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/threshold-prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/primary-contrasts.json"
test -s "${output_dir}/findings.md"
test -s "${output_dir}/research-taste-secondary.json"
test -s "${output_dir}/research-taste-findings.md"
test -s "${output_dir}/research-taste-human-calibration.jsonl"
test -s "${output_dir}/research-taste-human-calibration-key.json"
test -s "${output_dir}/research-taste-human-calibration-instructions.md"
