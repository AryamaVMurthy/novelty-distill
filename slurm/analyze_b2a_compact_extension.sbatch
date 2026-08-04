#!/bin/bash
# Analyze the predeclared random-1 sequence-KD extension against corrected K=4 artifacts.

#SBATCH --job-name=nd-analyze-b2a
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
output_dir="${evaluation_root}/compact-k4-seed17-b2a-extension-corrected-v2"

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

collector_inputs=(
  --input "A0=${evaluation_root}/A0-temporal-k4-from-k16-corrected-v2.json"
  --input "B2a=${evaluation_root}/B2a-tomato1k-seed17-temporal-k4-corrected-v2.json"
  --input "B2b=${evaluation_root}/B2b-tomato1k-seed17-temporal-k4-corrected-v2.json"
  --input "C1-best1=${evaluation_root}/C1-best1-tomato1k-seed17-temporal-k4-corrected-v2.json"
)
for input_index in 1 3 5 7; do
  test -s "${collector_inputs[input_index]#*=}"
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
  --config "${repo_dir}/configs/evaluation/b2a_compact_extension_contrasts.yaml" \
  --semantic-validity-policy "${repo_dir}/configs/evaluation/semantic_validity.yaml" \
  --output "${output_dir}/paired-contrasts.json" \
  --markdown "${output_dir}/findings.md"

test -s "${output_dir}/prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/threshold-prompt-metrics.jsonl.manifest.json"
test -s "${output_dir}/paired-contrasts.json"
test -s "${output_dir}/findings.md"
echo "b2a_extension_analysis=${output_dir}"
