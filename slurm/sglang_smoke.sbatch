#!/bin/bash
#SBATCH --job-name=nd-sglang
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_url="${REPO_URL:-https://github.com/AryamaVMurthy/novelty-distill.git}"
repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
model_path="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
model_revision="${MODEL_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
generation_config="${GENERATION_CONFIG:-configs/generation/smoke.yaml}"
input_limit="${INPUT_LIMIT:-1}"
output_name="${OUTPUT_NAME:-sglang-smoke}"
server_log="${scratch_root}/logs/sglang-${SLURM_JOB_ID}.log"
input_slice="${scratch_root}/tmp/generation-input-${SLURM_JOB_ID}.jsonl"
server_pid=""
if ! [[ "${input_limit}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INPUT_LIMIT must be a positive integer" >&2
  exit 1
fi

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
export HF_HOME="${scratch_root}/cache/huggingface"
export TORCH_HOME="${scratch_root}/cache/torch"
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
      if ! kill -0 "${server_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill -KILL "${server_pid}"
    fi
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT

mkdir -p \
  "${scratch_root}" \
  "${scratch_root}/cache/flashinfer-workspace" \
  "${scratch_root}/cache/torch-extensions" \
  "${scratch_root}/cache/triton" \
  "${scratch_root}/cache/cuda" \
  "${scratch_root}/cache/xdg" \
  "${scratch_root}/logs" \
  "${scratch_root}/tmp"
if [[ ! -d "${repo_dir}/.git" ]]; then
  git clone --branch "${repo_ref}" --single-branch "${repo_url}" "${repo_dir}"
else
  git -C "${repo_dir}" diff --quiet
  git -C "${repo_dir}" fetch origin "${repo_ref}"
  git -C "${repo_dir}" switch "${repo_ref}"
  git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  uv venv --python 3.12 "${venv_dir}"
fi
"${venv_dir}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/inference.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

"${venv_dir}/bin/python" -m sglang.launch_server \
  --model-path "${model_path}" \
  --revision "${model_revision}" \
  --served-model-name "${model_path}" \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.75 \
  --disable-piecewise-cuda-graph \
  >"${server_log}" 2>&1 &
server_pid="$!"

echo "stage=wait_for_server"
ready=0
server_ready_attempts="${SERVER_READY_ATTEMPTS:-360}"
for _attempt in $(seq 1 "${server_ready_attempts}"); do
  if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
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

input="${scratch_root}/data/tomato-open-train-1000.jsonl"
output_dir="${scratch_root}/generations/${output_name}/${SLURM_JOB_ID}"
if [[ ! -f "${input}" ]]; then
  echo "missing prepared smoke data: ${input}" >&2
  exit 1
fi
mkdir -p "${output_dir}"
head -n "${input_limit}" "${input}" >"${input_slice}"
echo "stage=generate input=${input_slice} output=${output_dir}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/generate_sglang.py" \
  --input "${input_slice}" \
  --output-dir "${output_dir}" \
  --config "${repo_dir}/${generation_config}" \
  --concurrency 1
"${venv_dir}/bin/python" "${repo_dir}/scripts/generate_sglang.py" \
  --input "${input_slice}" \
  --output-dir "${output_dir}" \
  --config "${repo_dir}/${generation_config}" \
  --concurrency 1
find "${output_dir}" -maxdepth 1 -type f -name '*.json' -print

"${venv_dir}/bin/python" -c 'import sglang, torch, transformers; print({"sglang": sglang.__version__, "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'
