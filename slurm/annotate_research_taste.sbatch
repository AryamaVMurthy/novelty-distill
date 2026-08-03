#!/bin/bash
# Resumably annotate one permanent generation namespace with research-taste labels.

#SBATCH --job-name=nd-taste
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

module load u22/cuda/12.4

generation_job_id="${GENERATION_JOB_ID:?set GENERATION_JOB_ID}"
generation_name="${GENERATION_NAME:-evaluation-student}"
generation_config="${GENERATION_CONFIG:?set GENERATION_CONFIG}"
prompts_path="${PROMPTS_PATH:-}"
prompts_name="${PROMPTS_NAME:-}"
taste_namespace="${TASTE_NAMESPACE:-research-taste}"
taste_concurrency="${TASTE_CONCURRENCY:-8}"
taste_attempts="${TASTE_ATTEMPTS:-3}"
for safe_value in "${generation_job_id}" "${generation_name}" "${taste_namespace}"; do
  if [[ -z "${safe_value}" || "${safe_value}" == */* || "${safe_value}" == *..* ]]; then
    echo "generation IDs, names, and namespaces must be path-safe" >&2
    exit 2
  fi
done
if [[ -z "${prompts_path}" && -z "${prompts_name}" ]]; then
  echo "set PROMPTS_PATH or PROMPTS_NAME" >&2
  exit 2
fi
if [[ -n "${prompts_name}" && ("${prompts_name}" == */* || "${prompts_name}" == *..*) ]]; then
  echo "PROMPTS_NAME must be a path-safe filename" >&2
  exit 2
fi
if ! [[ "${taste_concurrency}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TASTE_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
if ! [[ "${taste_attempts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TASTE_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
generation_dir="${scratch_root}/generations/${generation_name}/${generation_job_id}"
output_dir="${scratch_root}/${taste_namespace}/${generation_job_id}"
taste_config="${repo_dir}/configs/evaluation/research_taste.yaml"
annotator_model="Qwen/Qwen3-32B-FP8"
annotator_revision="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
server_log="${scratch_root}/logs/research-taste-${SLURM_JOB_ID}.log"
server_pid=""
port="${PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
if [[ -n "${prompts_name}" ]]; then
  prompts_path="${scratch_root}/data/${prompts_name}"
fi
if [[ "${generation_config}" != /* ]]; then
  generation_config="${repo_dir}/${generation_config}"
fi

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_LINK_MODE=copy
export HF_HOME="${scratch_root}/cache/huggingface"
export FLASHINFER_WORKSPACE_BASE="${scratch_root}/cache/flashinfer-workspace"
export XDG_CACHE_HOME="${scratch_root}/cache/xdg"
export TORCH_EXTENSIONS_DIR="${scratch_root}/cache/torch-extensions"
export TRITON_CACHE_DIR="${scratch_root}/cache/triton"
export CUDA_CACHE_PATH="${scratch_root}/cache/cuda"
export TMPDIR="${scratch_root}/tmp"

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    for _attempt in $(seq 1 10); do
      kill -0 "${server_pid}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT

mkdir -p "${scratch_root}/logs" "${scratch_root}/locks" "${scratch_root}/tmp"
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9
exec 8>"${scratch_root}/locks/venv-inference.lock"
flock -x 8
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/inference.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

set +e
"${venv_dir}/bin/python" "${repo_dir}/scripts/check_research_taste_status.py" \
  --generation-dir "${generation_dir}" \
  --generation-config "${generation_config}" \
  --output-dir "${output_dir}" \
  --taste-config "${taste_config}"
taste_status=$?
set -e
if [[ "${taste_status}" -eq 0 ]]; then
  echo "stage=already_complete output=${output_dir}"
  exit 0
fi
if [[ "${taste_status}" -ne 3 ]]; then
  echo "research-taste preflight failed with status ${taste_status}" >&2
  exit "${taste_status}"
fi

"${venv_dir}/bin/python" -m sglang.launch_server \
  --model-path "${annotator_model}" \
  --revision "${annotator_revision}" \
  --served-model-name "${annotator_model}" \
  --host 127.0.0.1 \
  --port "${port}" \
  --mem-fraction-static 0.90 \
  --disable-piecewise-cuda-graph \
  --enable-deterministic-inference \
  >"${server_log}" 2>&1 &
server_pid="$!"

ready=0
server_ready_attempts="${SERVER_READY_ATTEMPTS:-360}"
for _attempt in $(seq 1 "${server_ready_attempts}"); do
  if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -100 "${server_log}"
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" != 1 ]]; then
  tail -100 "${server_log}"
  exit 1
fi

"${venv_dir}/bin/python" "${repo_dir}/scripts/annotate_research_taste.py" \
  --generation-dir "${generation_dir}" \
  --generation-config "${generation_config}" \
  --prompts "${prompts_path}" \
  --output-dir "${output_dir}" \
  --taste-config "${taste_config}" \
  --concurrency "${taste_concurrency}" \
  --attempts "${taste_attempts}" \
  --base-url "http://127.0.0.1:${port}"
