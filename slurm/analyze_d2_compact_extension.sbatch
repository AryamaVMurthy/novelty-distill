#!/bin/bash
# Collect the seven-method K=4 matrix and run the preregistered D2 extension contrasts.

#SBATCH --job-name=nd-analyze-d2
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"
evaluation_root="${scratch_root}/evaluations/tomato"
output_dir="${evaluation_root}/compact-k4-seed17-d2-extension"

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

collector_inputs=(--input "A0=${evaluation_root}/A0-temporal-k4-from-k16.json")
for baseline_id in B1 B2b C1-best1 C2-best1 D1 D2; do
  eval_id="${baseline_id}-tomato1k-seed17-temporal-k4"
  test -s "${evaluation_root}/${eval_id}.json"
  collector_inputs+=(--input "${baseline_id}=${evaluation_root}/${eval_id}.json")
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
  --config "${repo_dir}/configs/evaluation/d2_compact_extension_contrasts.yaml" \
  --output "${output_dir}/paired-contrasts.json" \
  --markdown "${output_dir}/findings.md"

test -s "${output_dir}/prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/threshold-prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/paired-contrasts.json"
test -s "${output_dir}/findings.md"
echo "d2_extension_analysis=${output_dir}"
