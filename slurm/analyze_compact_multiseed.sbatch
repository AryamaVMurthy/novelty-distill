#!/bin/bash
# Analyze the clean six-method compact K=4 matrix over checkpoint seeds 17/29/43.

#SBATCH --job-name=nd-analyze-compact-3seed
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
train_seeds="${TRAIN_SEEDS:-17,29,43}"
compact_methods="${COMPACT_METHODS:-B2a,B2b,C1-best1,C2-best1,D1,D2}"
if [[ "${train_seeds}" != "17,29,43" ]]; then
  echo "TRAIN_SEEDS must be the frozen 17,29,43 replication" >&2
  exit 2
fi
if [[ "${compact_methods}" != "B2a,B2b,C1-best1,C2-best1,D1,D2" ]]; then
  echo "COMPACT_METHODS does not match the frozen clean baseline set" >&2
  exit 2
fi

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"
evaluation_root="${scratch_root}/evaluations/tomato"
output_dir="${evaluation_root}/compact-k4-three-seed-corrected-v2"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9

export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${scratch_root}/cache/uv-analysis"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
mkdir -p "${scratch_root}/locks" "${output_dir}"
exec 8>"${scratch_root}/locks/venv-data.lock"
flock -x 8
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/data.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

a0_path="${evaluation_root}/A0-temporal-k4-from-k16-corrected-v2.json"
test -s "${a0_path}"
analysis_inputs=()
IFS=',' read -r -a methods <<<"${compact_methods}"
IFS=',' read -r -a seeds <<<"${train_seeds}"
for method in "${methods[@]}"; do
  for seed in "${seeds[@]}"; do
    path="${evaluation_root}/${method}-tomato1k-seed${seed}-temporal-k4-corrected-v2.json"
    test -s "${path}"
    analysis_inputs+=(--input "${method}:${seed}=${path}")
  done
done

"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_multiseed_contrasts.py" \
  "${analysis_inputs[@]}" \
  --control "A0=${a0_path}" \
  --config "${repo_dir}/configs/evaluation/compact_multiseed_contrasts.yaml" \
  --output "${output_dir}/checkpoint-seed-contrasts.json" \
  --markdown "${output_dir}/findings.md"

test -s "${output_dir}/checkpoint-seed-contrasts.json"
test -s "${output_dir}/findings.md"
echo "completed_compact_multiseed_analysis=${output_dir}"
