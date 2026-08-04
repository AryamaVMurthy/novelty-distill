#!/bin/bash
# Globally validate a sharded generation run before releasing downstream work.

#SBATCH --job-name=nd-validate-generation
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
generation_name="${GENERATION_NAME:-teacher}"
generation_id="${GENERATION_ID:-teacher-1k-v1}"
generation_config="${GENERATION_CONFIG:-configs/generation/teacher.yaml}"
input_path="${INPUT_PATH:-}"
input_name="${INPUT_NAME:-}"
expected_prompts="${EXPECTED_PROMPTS:-1000}"
served_artifact_identity="${SERVED_ARTIFACT_IDENTITY:-}"
for value in "${generation_name}" "${generation_id}"; do
  if [[ -z "${value}" || "${value}" == */* || "${value}" == *..* ]]; then
    echo "generation names must be path-safe" >&2
    exit 2
  fi
done
if [[ -n "${input_name}" && ("${input_name}" == */* || "${input_name}" == *..*) ]]; then
  echo "INPUT_NAME must be a path-safe filename" >&2
  exit 2
fi
if ! [[ "${expected_prompts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_PROMPTS must be a positive integer" >&2
  exit 2
fi

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${input_name}" ]]; then
  input_path="${scratch_root}/data/${input_name}"
elif [[ -z "${input_path}" ]]; then
  input_path="${scratch_root}/data/tomato-open-train-${expected_prompts}.jsonl"
fi
if [[ "${generation_config}" != /* ]]; then
  generation_config="${repo_dir}/${generation_config}"
fi
output_dir="${scratch_root}/generations/${generation_name}/${generation_id}"
input_slice="${scratch_root}/tmp/generation-validation-${SLURM_JOB_ID}.jsonl"

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy

mkdir -p "${scratch_root}/locks" "${scratch_root}/tmp"
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9
exec 8>"${scratch_root}/locks/venv-inference.lock"
flock -x 8
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

test -f "${input_path}"
head -n "${expected_prompts}" "${input_path}" >"${input_slice}"
test "$(wc -l <"${input_slice}")" -eq "${expected_prompts}"
artifact_args=()
if [[ -n "${served_artifact_identity}" ]]; then
  artifact_args=(--served-artifact-identity "${served_artifact_identity}")
fi
"${venv_dir}/bin/python" "${repo_dir}/scripts/check_generation_status.py" \
  --input "${input_slice}" \
  --output-dir "${output_dir}" \
  --config "${generation_config}" \
  --num-shards 1 \
  --shard-index 0 \
  "${artifact_args[@]}"
test "$(find "${output_dir}" -maxdepth 1 -type f -name '*.json' | wc -l)" \
  -eq "${expected_prompts}"
echo "validated_generation=${generation_name}/${generation_id} prompts=${expected_prompts}"
