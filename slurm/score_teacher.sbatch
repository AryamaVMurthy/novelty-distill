#!/bin/bash
#SBATCH --job-name=nd-score-teacher
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

module load u22/cuda/12.4

generation_job_id="${GENERATION_JOB_ID:?set GENERATION_JOB_ID to the teacher SGLang job}"
repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
judge_model="Qwen/Qwen3-32B-FP8"
judge_revision="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
server_log="${scratch_root}/logs/judge-${SLURM_JOB_ID}.log"
server_pid=""

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

mkdir -p "${scratch_root}/logs" "${scratch_root}/tmp"
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/inference.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

"${venv_dir}/bin/python" -m sglang.launch_server \
  --model-path "${judge_model}" \
  --revision "${judge_revision}" \
  --served-model-name "${judge_model}" \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.90 \
  --disable-piecewise-cuda-graph \
  >"${server_log}" 2>&1 &
server_pid="$!"

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

"${venv_dir}/bin/python" "${repo_dir}/scripts/score_teacher_generations.py" \
  --generation-dir "${scratch_root}/generations/teacher-smoke/${generation_job_id}" \
  --generation-config "${repo_dir}/configs/generation/teacher.yaml" \
  --prompts "${scratch_root}/data/tomato-open-train-1000.jsonl" \
  --output-dir "${scratch_root}/teacher-scores/${generation_job_id}" \
  --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml"
