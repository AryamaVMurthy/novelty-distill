#!/bin/bash
# Globally validate every score shard before clustering teacher generations.

#SBATCH --job-name=nd-validate-scores
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
generation_name="${GENERATION_NAME:-teacher}"
generation_id="${GENERATION_ID:-teacher-1k-v1}"
generation_config="${GENERATION_CONFIG:-configs/generation/teacher.yaml}"
score_namespace="${SCORE_NAMESPACE:-teacher-scores}"
expected_prompts="${EXPECTED_PROMPTS:-1000}"
for value in "${generation_name}" "${generation_id}" "${score_namespace}"; do
  if [[ -z "${value}" || "${value}" == */* || "${value}" == *..* ]]; then
    echo "generation and score names must be path-safe" >&2
    exit 2
  fi
done
if ! [[ "${expected_prompts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_PROMPTS must be a positive integer" >&2
  exit 2
fi

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
if [[ "${generation_config}" != /* ]]; then
  generation_config="${repo_dir}/${generation_config}"
fi
generation_dir="${scratch_root}/generations/${generation_name}/${generation_id}"
score_dir="${scratch_root}/${score_namespace}/${generation_id}"

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy

mkdir -p "${scratch_root}/locks"
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9
exec 8>"${scratch_root}/locks/venv-inference.lock"
flock -x 8
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

test "$(find "${generation_dir}" -maxdepth 1 -type f -name '*.json' | wc -l)" \
  -eq "${expected_prompts}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/check_score_status.py" \
  --generation-dir "${generation_dir}" \
  --generation-config "${generation_config}" \
  --output-dir "${score_dir}" \
  --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml" \
  --num-shards 1 \
  --shard-index 0
test "$(find "${score_dir}" -maxdepth 1 -type f -name '*.json' | wc -l)" \
  -eq "${expected_prompts}"
echo "validated_scores=${score_namespace}/${generation_id} prompts=${expected_prompts}"
