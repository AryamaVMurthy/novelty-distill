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

calibration_run_id="${CALIBRATION_RUN_ID:-}"
generation_job_id="${GENERATION_JOB_ID:-}"
generation_name="${GENERATION_NAME:-teacher-smoke}"
generation_config="${GENERATION_CONFIG:-configs/generation/teacher.yaml}"
prompts_path="${PROMPTS_PATH:-}"
score_namespace="${SCORE_NAMESPACE:-teacher-scores}"
score_concurrency="${SCORE_CONCURRENCY:-8}"
if [[ -z "${calibration_run_id}" && -z "${generation_job_id}" ]]; then
  echo "set GENERATION_JOB_ID or CALIBRATION_RUN_ID" >&2
  exit 2
fi
if [[ -n "${calibration_run_id}" && ("${calibration_run_id}" == */* || "${calibration_run_id}" == *..*) ]]; then
  echo "CALIBRATION_RUN_ID must be a path-safe name" >&2
  exit 2
fi
if [[ "${score_namespace}" == */* || "${score_namespace}" == *..* ]]; then
  echo "SCORE_NAMESPACE must be a path-safe name" >&2
  exit 2
fi
if ! [[ "${score_concurrency}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SCORE_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
if [[ -z "${prompts_path}" ]]; then
  prompts_path="${scratch_root}/data/tomato-open-train-1000.jsonl"
fi
if [[ "${generation_config}" != /* ]]; then
  generation_config="${repo_dir}/${generation_config}"
fi
judge_model="Qwen/Qwen3-32B-FP8"
judge_revision="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
server_log="${scratch_root}/logs/judge-${SLURM_JOB_ID}.log"
server_pid=""
port="${PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

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

if [[ -n "${calibration_run_id}" ]]; then
  calibration_root="${scratch_root}/generations/teacher-calibration/${calibration_run_id}"
  score_root="${scratch_root}/teacher-calibration-scores/${calibration_run_id}"
  test -f "${calibration_root}/study-manifest.json"
  conditions=0
  for condition_dir in "${calibration_root}"/*; do
    [[ -d "${condition_dir}/shards" ]] || continue
    condition="$(basename "${condition_dir}")"
    "${venv_dir}/bin/python" "${repo_dir}/scripts/score_teacher_generations.py" \
      --generation-dir "${condition_dir}/shards" \
      --generation-config "${condition_dir}/generation-config.json" \
      --prompts "${scratch_root}/data/tomato-open-calibration-8.jsonl" \
      --output-dir "${score_root}/${condition}" \
      --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml" \
      --concurrency "${score_concurrency}" \
      --base-url "http://127.0.0.1:${port}"
    conditions=$((conditions + 1))
  done
  test "${conditions}" -gt 0
  echo "scored_calibration_conditions=${conditions}"
else
  "${venv_dir}/bin/python" "${repo_dir}/scripts/score_teacher_generations.py" \
    --generation-dir "${scratch_root}/generations/${generation_name}/${generation_job_id}" \
    --generation-config "${generation_config}" \
    --prompts "${prompts_path}" \
    --output-dir "${scratch_root}/${score_namespace}/${generation_job_id}" \
    --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml" \
    --concurrency "${score_concurrency}" \
    --base-url "http://127.0.0.1:${port}"
fi
