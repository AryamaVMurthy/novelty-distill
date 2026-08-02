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
eval_id="${EVAL_ID:-base-qwen3-4b}"
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"
model_revision="${MODEL_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"
model_dtype="${MODEL_DTYPE:-auto}"
served_model="novelty-model"
base_served_model="${served_model}"
lora_name="${LORA_NAME:-}"
lora_path="${LORA_PATH:-}"
domain="${HYPOSPACE_DOMAIN:-causal}"
eval_limit="${EVAL_LIMIT:-}"
num_generations="${NUM_GENERATIONS:-2}"
repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
inference_venv="${scratch_root}/venvs/inference"
server_log="${scratch_root}/logs/eval-sglang-${SLURM_JOB_ID}.log"
server_pid=""
port="${PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
if [[ "${model_dtype}" != auto && "${model_dtype}" != bfloat16 && "${model_dtype}" != float16 ]]; then
  echo "MODEL_DTYPE must be auto, bfloat16, or float16" >&2
  exit 2
fi
if [[ -z "${eval_id}" || "${eval_id}" == */* || "${eval_id}" == *..* ]]; then
  echo "EVAL_ID must be a non-empty path-safe name" >&2
  exit 2
fi
if [[ -z "${eval_limit}" ]]; then
  if [[ "${eval_suite}" == noveltybench ]]; then
    eval_limit=2
  elif [[ "${eval_suite}" == hypospace ]]; then
    eval_limit=1
  else
    echo "unsupported EVAL_SUITE=${eval_suite}" >&2
    exit 2
  fi
fi
for value in "${eval_limit}" "${num_generations}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_LIMIT and NUM_GENERATIONS must be positive integers" >&2
    exit 2
  fi
done
if [[ -n "${lora_name}" || -n "${lora_path}" ]]; then
  if [[ -z "${lora_name}" || -z "${lora_path}" ]]; then
    echo "LORA_NAME and LORA_PATH must be set together" >&2
    exit 2
  fi
  if [[ "${lora_name}" == */* || "${lora_name}" == *..* || ! -d "${lora_path}" ]]; then
    echo "LORA_NAME must be path-safe and LORA_PATH must be a directory" >&2
    exit 2
  fi
  if [[ "${lora_name}" != "${served_model}" ]]; then
    echo "official evaluation LORA_NAME must be ${served_model}" >&2
    exit 2
  fi
  base_served_model="novelty-base"
fi
result_root="${scratch_root}/evaluations/official/${eval_id}"

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_LINK_MODE=copy
export HF_HOME="${scratch_root}/cache/huggingface"
export FLASHINFER_WORKSPACE_BASE="${scratch_root}/cache/flashinfer-workspace"
export XDG_CACHE_HOME="${scratch_root}/cache/xdg"
export TORCH_EXTENSIONS_DIR="${scratch_root}/cache/torch-extensions"
export TRITON_CACHE_DIR="${scratch_root}/cache/triton"
export TVM_FFI_CACHE_DIR="${scratch_root}/cache/tvm-ffi"
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

mkdir -p \
  "${scratch_root}/logs" \
  "${scratch_root}/tmp" \
  "${scratch_root}/evaluations" \
  "${TVM_FFI_CACHE_DIR}"
mkdir -p "${scratch_root}/locks"
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
flock -u 9
exec 8>"${scratch_root}/locks/venv-inference.lock"
flock -x 8
uv pip sync --python "${inference_venv}/bin/python" "${repo_dir}/environments/inference.lock"
uv pip install --python "${inference_venv}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

artifact_path=""
if [[ -n "${lora_path}" ]]; then
  artifact_path="${lora_path}"
elif [[ -d "${model_path}" ]]; then
  artifact_path="${model_path}"
fi
if [[ -n "${artifact_path}" ]]; then
  model_identity="sha256:$(${inference_venv}/bin/python "${repo_dir}/scripts/hash_model_artifact.py" --path "${artifact_path}")"
else
  model_identity="hf:${model_path}@${model_revision}"
fi
if [[ "${eval_suite}" == noveltybench ]]; then
  summary_path="${result_root}/noveltybench/summary.json"
  summary_domain=()
elif [[ "${eval_suite}" == hypospace ]]; then
  case "${domain}" in causal|3d|boolean) ;; *) echo "unsupported HYPOSPACE_DOMAIN=${domain}" >&2; exit 2 ;; esac
  summary_path="${result_root}/hypospace/${domain}/summary.json"
  summary_domain=(--domain "${domain}")
else
  echo "unsupported EVAL_SUITE=${eval_suite}" >&2
  exit 2
fi
set +e
"${inference_venv}/bin/python" "${repo_dir}/scripts/check_official_evaluation_status.py" \
  --summary "${summary_path}" \
  --eval-id "${eval_id}" \
  --suite "${eval_suite}" \
  "${summary_domain[@]}" \
  --model-identity "${model_identity}" \
  --expected-samples "${eval_limit}" \
  --num-generations "${num_generations}"
official_status=$?
set -e
if [[ "${official_status}" -eq 0 ]]; then
  echo "stage=already_complete output=${summary_path}"
  exit 0
fi
if [[ "${official_status}" -ne 3 ]]; then
  echo "official evaluation preflight failed with status ${official_status}" >&2
  exit "${official_status}"
fi

server_args=(
  --model-path "${model_path}"
  --served-model-name "${base_served_model}"
  --dtype "${model_dtype}"
  --host 127.0.0.1
  --port "${port}"
  --mem-fraction-static 0.55
  --disable-piecewise-cuda-graph
  --enable-deterministic-inference
)
if [[ -n "${model_revision}" ]]; then
  server_args+=(--revision "${model_revision}")
fi
if [[ -n "${lora_name}" ]]; then
  server_args+=(--lora-paths "${lora_name}=${lora_path}")
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
    evaluation_dir="${result_root}/noveltybench"
    mkdir -p "${evaluation_dir}"
    export UV_PROJECT_ENVIRONMENT="${scratch_root}/venvs/noveltybench"
    export SGLANG_BASE_URL="http://127.0.0.1:${port}/v1"
    export SGLANG_API_KEY=local
    (
      cd "${package_dir}"
      exec 7>"${scratch_root}/locks/venv-noveltybench.lock"
      flock -x 7
      uv sync --frozen
      uv pip install --python "${scratch_root}/venvs/noveltybench/bin/python" \
        openai==2.26.0 distro==1.9.0 jiter==0.13.0 torch==2.9.1
      flock -u 7
      uv run --frozen --no-sync inspect eval inspect_evals/novelty_bench \
        --model "sglang/${served_model}" \
        --model-base-url "${SGLANG_BASE_URL}" \
        -M api_key=local \
        -T dataset_split="${NOVELTY_SPLIT:-curated}" \
        -T num_generations="${num_generations}" \
        -T shuffle=false \
        -T quality_model=small \
        -T quality_model_device=cuda \
        -T partition_model_device=cuda \
        --limit "${eval_limit}" \
        --max-connections "${MAX_CONNECTIONS:-2}" \
        --seed "${SEED:-17}" \
        --temperature "${TEMPERATURE:-0.7}" \
        --reasoning-effort none \
        --top-p "${TOP_P:-0.9}" \
        --max-tokens "${MAX_TOKENS:-512}" \
        --log-dir "${evaluation_dir}"
      PYTHONPATH="${repo_dir}/src" \
        "${scratch_root}/venvs/noveltybench/bin/python" \
        "${repo_dir}/scripts/summarize_noveltybench.py" \
          --log-dir "${evaluation_dir}" \
          --output "${summary_path}" \
          --eval-id "${eval_id}" \
          --model-identity "${model_identity}" \
          --expected-samples "${eval_limit}" \
          --num-generations "${num_generations}"
    )
    ;;
  hypospace)
    official_root="${scratch_root}/official/hypospace"
    hypospace_venv="${scratch_root}/venvs/hypospace"
    evaluation_dir="${result_root}/hypospace/${domain}"
    test "$(git -C "${official_root}" rev-parse HEAD)" = \
      "c69e9318577b34b5b896996571aefd4ba6053f58"
    exec 7>"${scratch_root}/locks/venv-hypospace.lock"
    flock -x 7
    uv pip sync --python "${hypospace_venv}/bin/python" \
      "${repo_dir}/environments/hypospace.lock"
    uv pip install --python "${hypospace_venv}/bin/python" --no-deps -e "${repo_dir}"
    flock -u 7
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
        --n-samples "${eval_limit}" \
        --n-queries "${num_generations}" \
        --query-multiplier 1.0 \
        --seed 33550336
    )
    "${inference_venv}/bin/python" "${repo_dir}/scripts/summarize_hypospace.py" \
      --input "${evaluation_dir}/${SLURM_JOB_ID}.json" \
      --output "${summary_path}" \
      --eval-id "${eval_id}" \
      --domain "${domain}" \
      --model-identity "${model_identity}" \
      --expected-samples "${eval_limit}" \
      --num-generations "${num_generations}"
    ;;
  *)
    echo "unsupported EVAL_SUITE=${eval_suite}" >&2
    exit 2
    ;;
esac
