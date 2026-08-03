#!/bin/bash
# Analyze the secondary matched 1x/4x exposure matrix after all primary and 4x outputs exist.

#SBATCH --job-name=nd-analyze-exposure
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
score_root="${scratch_root}/evaluation-scores"
output_dir="${evaluation_root}/analysis-exposure4x-seed17"
primary_methods=(B2a B2b B2c B3 B4)
sensitivity_methods=(B2a-4x B2b-4x B2c-4x B3-4x B4-4x)

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9

export UV_CACHE_DIR="${scratch_root}/cache/uv-analysis"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
mkdir -p "${scratch_root}/locks" "${output_dir}"
exec 8>"${scratch_root}/locks/venv-data.lock"
flock -x 8
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/data.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

collector_inputs=()
taste_inputs=(
  --input "A1=${taste_root}/A1-temporal-k16-seed17000"
  --input "A3=${taste_root}/A3-temporal-k1"
)
geometry_inputs=(
  --input "A0=${score_root}/A0-temporal-k16-seed17000"
  --input "A1=${score_root}/A1-temporal-k16-seed17000"
  --input "A3=${score_root}/A3-temporal-k1"
)
wandb_inputs=()
for method in "${primary_methods[@]}"; do
  eval_id="${method}-tomato1k-seed17-temporal-k16"
  test -s "${evaluation_root}/${eval_id}.json"
  test -d "${taste_root}/${eval_id}"
  collector_inputs+=(--input "${method}=${evaluation_root}/${eval_id}.json")
  taste_inputs+=(--input "${method}=${taste_root}/${eval_id}")
  test -d "${score_root}/${eval_id}"
  geometry_inputs+=(--input "${method}=${score_root}/${eval_id}")
  wandb_inputs+=(--input "${evaluation_root}/${eval_id}.json")
done

audit_inputs=()
for method in "${sensitivity_methods[@]}"; do
  run_name="${method}-tomato1k-exposure4x-seed17"
  eval_id="${run_name}-temporal-k16"
  metadata_path="${scratch_root}/checkpoints/${run_name}/run_metadata.json"
  test -s "${metadata_path}"
  test -s "${evaluation_root}/${eval_id}.json"
  test -d "${taste_root}/${eval_id}"
  audit_inputs+=(--input "${method}=${metadata_path}")
  collector_inputs+=(--input "${method}=${evaluation_root}/${eval_id}.json")
  taste_inputs+=(--input "${method}=${taste_root}/${eval_id}")
  test -d "${score_root}/${eval_id}"
  geometry_inputs+=(--input "${method}=${score_root}/${eval_id}")
  wandb_inputs+=(--input "${metadata_path}" --input "${evaluation_root}/${eval_id}.json")
done

"${venv_dir}/bin/python" "${repo_dir}/scripts/audit_training_matrix.py" \
  --registry "${repo_dir}/configs/exposure_sensitivity_baselines.yaml" \
  "${audit_inputs[@]}" \
  --output "${output_dir}/training-matrix-audit.json"
"${venv_dir}/bin/python" "${repo_dir}/scripts/collect_evaluation_metrics.py" \
  "${collector_inputs[@]}" \
  --output "${output_dir}/prompt-metrics.jsonl" \
  --threshold-output "${output_dir}/threshold-prompt-metrics.jsonl"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_contrast_matrix.py" \
  --input "${output_dir}/prompt-metrics.jsonl" \
  --threshold-input "${output_dir}/threshold-prompt-metrics.jsonl" \
  --config "${repo_dir}/configs/evaluation/exposure_sensitivity_contrasts.yaml" \
  --output "${output_dir}/secondary-contrasts.json" \
  --markdown "${output_dir}/findings.md"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_research_taste.py" \
  "${taste_inputs[@]}" \
  --human-method A3 \
  --teacher-method A1 \
  --config "${repo_dir}/configs/evaluation/research_taste.yaml" \
  --output "${output_dir}/research-taste-secondary.json" \
  --markdown "${output_dir}/research-taste-findings.md"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_same_prompt_geometry.py" \
  "${geometry_inputs[@]}" \
  --embedding-cache-dir "${scratch_root}/cache/embeddings" \
  --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml" \
  --analysis-config "${repo_dir}/configs/evaluation/same_prompt_geometry.yaml" \
  --output "${output_dir}/same-prompt-geometry-secondary.json" \
  --markdown "${output_dir}/same-prompt-geometry-findings.md"

wandb_inputs+=(
  --input "${output_dir}/training-matrix-audit.json"
  --input "${output_dir}/secondary-contrasts.json"
  --input "${output_dir}/research-taste-secondary.json"
  --input "${output_dir}/same-prompt-geometry-secondary.json"
)
"${venv_dir}/bin/python" "${repo_dir}/scripts/export_wandb_snapshot.py" \
  "${wandb_inputs[@]}" \
  --manifest "${output_dir}/wandb-export-manifest.json" \
  --run-dir "${scratch_root}/wandb"

test -s "${output_dir}/training-matrix-audit.json"
test -s "${output_dir}/secondary-contrasts.json"
test -s "${output_dir}/findings.md"
test -s "${output_dir}/research-taste-secondary.json"
test -s "${output_dir}/research-taste-findings.md"
test -s "${output_dir}/same-prompt-geometry-secondary.json"
test -s "${output_dir}/same-prompt-geometry-findings.md"
test -s "${output_dir}/wandb-export-manifest.json"
