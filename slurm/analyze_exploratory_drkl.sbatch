#!/bin/bash
# Analyze DRKL only after the frozen primary analysis and matched artifacts pass.

#SBATCH --job-name=nd-analyze-drkl
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
output_dir="${evaluation_root}/analysis-seed17/exploratory-drkl"

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

methods=(C2-best1 C2-diverse4 F1-best1 F1-diverse4)
collector_inputs=()
for method in "${methods[@]}"; do
  if [[ "${method}" == F1-* ]]; then
    eval_id="${method}-exploratory-drkl-tomato1k-seed17-temporal-k16"
  else
    eval_id="${method}-tomato1k-seed17-temporal-k16"
  fi
  test -s "${evaluation_root}/${eval_id}.json"
  collector_inputs+=(--input "${method}=${evaluation_root}/${eval_id}.json")
done

matrix_path="${output_dir}/prompt-metrics.jsonl"
threshold_matrix_path="${output_dir}/threshold-prompt-metrics.jsonl"
"${venv_dir}/bin/python" "${repo_dir}/scripts/collect_evaluation_metrics.py" \
  "${collector_inputs[@]}" \
  --output "${matrix_path}" \
  --threshold-output "${threshold_matrix_path}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_contrast_matrix.py" \
  --input "${matrix_path}" \
  --threshold-input "${threshold_matrix_path}" \
  --config "${repo_dir}/configs/evaluation/exploratory_drkl_contrasts.yaml" \
  --output "${output_dir}/contrasts.json" \
  --markdown "${output_dir}/findings.md"

taste_inputs=(
  --input "A3=${taste_root}/A3-temporal-k1"
  --input "A1=${taste_root}/A1-temporal-k16-seed17000"
  --input "C2-best1=${taste_root}/C2-best1-tomato1k-seed17-temporal-k16"
  --input "C2-diverse4=${taste_root}/C2-diverse4-tomato1k-seed17-temporal-k16"
  --input "F1-best1=${taste_root}/F1-best1-exploratory-drkl-tomato1k-seed17-temporal-k16"
  --input "F1-diverse4=${taste_root}/F1-diverse4-exploratory-drkl-tomato1k-seed17-temporal-k16"
)
for value in "${taste_inputs[@]}"; do
  if [[ "${value}" == *=* ]]; then
    test -d "${value#*=}"
  fi
done
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_research_taste.py" \
  "${taste_inputs[@]}" \
  --human-method A3 \
  --teacher-method A1 \
  --config "${repo_dir}/configs/evaluation/research_taste.yaml" \
  --output "${output_dir}/research-taste-secondary.json" \
  --markdown "${output_dir}/research-taste-findings.md"

wandb_inputs=(
  --input "${output_dir}/contrasts.json"
  --input "${output_dir}/research-taste-secondary.json"
)
for method in "${methods[@]}"; do
  if [[ "${method}" == F1-* ]]; then
    run_name="${method}-exploratory-drkl-tomato1k-seed17"
    eval_id="${run_name}-temporal-k16"
  else
    run_name="${method}-tomato1k-seed17"
    eval_id="${run_name}-temporal-k16"
  fi
  metadata_path="${scratch_root}/checkpoints/${run_name}/run_metadata.json"
  test -s "${metadata_path}"
  wandb_inputs+=(
    --input "${metadata_path}"
    --input "${evaluation_root}/${eval_id}.json"
  )
done
"${venv_dir}/bin/python" "${repo_dir}/scripts/export_wandb_snapshot.py" \
  "${wandb_inputs[@]}" \
  --manifest "${output_dir}/wandb-export-manifest.json" \
  --run-dir "${scratch_root}/wandb"

test -s "${output_dir}/prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/threshold-prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/contrasts.json"
test -s "${output_dir}/findings.md"
test -s "${output_dir}/research-taste-secondary.json"
test -s "${output_dir}/research-taste-findings.md"
test -s "${output_dir}/wandb-export-manifest.json"
