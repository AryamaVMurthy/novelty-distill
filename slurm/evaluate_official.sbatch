#!/bin/bash
#SBATCH --job-name=nd-eval-official
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

module load u22/cuda/12.4

eval_suite="${EVAL_SUITE:-noveltybench}"
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"
model_revision="${MODEL_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"
served_model="novelty-model"
repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
inference_venv="${scratch_root}/venvs/inference"
server_log="${scratch_root}/logs/eval-sglang-${SLURM_JOB_ID}.log"
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

mkdir -p "${scratch_root}/logs" "${scratch_root}/tmp" "${scratch_root}/evaluations"
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
uv pip sync --python "${inference_venv}/bin/python" "${repo_dir}/environments/inference.lock"

server_args=(
  --model-path "${model_path}"
  --served-model-name "${served_model}"
  --host 127.0.0.1
  --port "${port}"
  --mem-fraction-static 0.55
  --disable-piecewise-cuda-graph
)
if [[ -n "${model_revision}" ]]; then
  server_args+=(--revision "${model_revision}")
fi
"${inference_venv}/bin/python" -m sglang.launch_server "${server_args[@]}" \
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

case "${eval_suite}" in
  noveltybench)
    inspect_root="${scratch_root}/official/inspect_evals"
    package_dir="${inspect_root}/packages/novelty_bench"
    test "$(git -C "${inspect_root}" rev-parse HEAD)" = \
      "6a35510e530f236fd1dbcd9df888f01937c8494a"
    export UV_PROJECT_ENVIRONMENT="${scratch_root}/venvs/noveltybench"
    export SGLANG_BASE_URL="http://127.0.0.1:${port}/v1"
    export SGLANG_API_KEY=local
    (
      cd "${package_dir}"
      uv sync --frozen
      uv pip install --python "${scratch_root}/venvs/noveltybench/bin/python" \
        openai==2.26.0 distro==1.9.0 jiter==0.13.0 torch==2.9.1
      uv run --frozen --no-sync inspect eval inspect_evals/novelty_bench \
        --model "sglang/${served_model}" \
        --model-base-url "${SGLANG_BASE_URL}" \
        -M api_key=local \
        -T dataset_split="${NOVELTY_SPLIT:-curated}" \
        -T num_generations="${NUM_GENERATIONS:-2}" \
        -T shuffle=false \
        -T quality_model=small \
        -T quality_model_device=cuda \
        -T partition_model_device=cuda \
        --limit "${EVAL_LIMIT:-2}" \
        --max-connections "${MAX_CONNECTIONS:-2}" \
        --seed "${SEED:-17}" \
        --temperature "${TEMPERATURE:-0.7}" \
        --reasoning-effort none \
        --top-p "${TOP_P:-0.9}" \
        --max-tokens "${MAX_TOKENS:-512}" \
        --log-dir "${scratch_root}/evaluations/noveltybench"
    )
    ;;
  hypospace)
    domain="${HYPOSPACE_DOMAIN:-causal}"
    official_root="${scratch_root}/official/hypospace"
    hypospace_venv="${scratch_root}/venvs/hypospace"
    evaluation_dir="${scratch_root}/evaluations/hypospace/${domain}"
    test "$(git -C "${official_root}" rev-parse HEAD)" = \
      "c69e9318577b34b5b896996571aefd4ba6053f58"
    uv pip sync --python "${hypospace_venv}/bin/python" \
      "${repo_dir}/environments/hypospace.lock"
    uv pip install --python "${hypospace_venv}/bin/python" --no-deps -e "${repo_dir}"
    case "${domain}" in
      causal) dataset="${official_root}/causal/datasets/node03/n3_all_observations.json" ;;
      3d) dataset="${official_root}/3d/datasets/3d_grid3_h3.json" ;;
      boolean) dataset="${official_root}/boolean/datasets/boolean_2var.json" ;;
      *) echo "unsupported HYPOSPACE_DOMAIN=${domain}" >&2; exit 2 ;;
    esac
    mkdir -p "${evaluation_dir}"
    (
      cd "${evaluation_dir}"
      "${hypospace_venv}/bin/python" "${repo_dir}/scripts/run_hypospace_official.py" \
        --repository "${official_root}" \
        --domain "${domain}" \
        --checkpoint-dir "${evaluation_dir}/checkpoints" \
        --output "${evaluation_dir}/${SLURM_JOB_ID}.json" \
        --base-url "http://127.0.0.1:${port}/v1" \
        --max-tokens "${MAX_TOKENS:-512}" \
        -- \
        --dataset "${dataset}" \
        --config "${repo_dir}/configs/evaluation/hypospace_sglang.yaml" \
        --n-samples "${EVAL_LIMIT:-1}" \
        --n-queries "${NUM_GENERATIONS:-2}" \
        --query-multiplier 1.0 \
        --seed 33550336
    )
    ;;
  *)
    echo "unsupported EVAL_SUITE=${eval_suite}" >&2
    exit 2
    ;;
esac
