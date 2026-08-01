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
server_log="${scratch_root}/sglang-smoke-server.log"
server_pid=""

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export HF_HOME="${scratch_root}/cache/huggingface"
export TORCH_HOME="${scratch_root}/cache/torch"

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT

mkdir -p "${scratch_root}" "${scratch_root}/cache"
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
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/inference.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

"${venv_dir}/bin/python" -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --served-model-name Qwen/Qwen3-1.7B \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.75 \
  >"${server_log}" 2>&1 &
server_pid="$!"

ready=0
for _attempt in $(seq 1 120); do
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
output_dir="${scratch_root}/generations/sglang-smoke"
if [[ ! -f "${input}" ]]; then
  echo "missing prepared smoke data: ${input}" >&2
  exit 1
fi
mkdir -p "${output_dir}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/generate_sglang.py" \
  --input <(head -n 1 "${input}") \
  --output-dir "${output_dir}" \
  --config "${repo_dir}/configs/generation/smoke.yaml" \
  --concurrency 1
"${venv_dir}/bin/python" "${repo_dir}/scripts/generate_sglang.py" \
  --input <(head -n 1 "${input}") \
  --output-dir "${output_dir}" \
  --config "${repo_dir}/configs/generation/smoke.yaml" \
  --concurrency 1
find "${output_dir}" -maxdepth 1 -type f -name '*.json' -print

"${venv_dir}/bin/python" -c 'import sglang, torch, transformers; print({"sglang": sglang.__version__, "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'
