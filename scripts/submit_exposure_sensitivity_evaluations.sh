#!/bin/bash
# Submit temporal evaluation and research-taste chains for the 4x-exposure matrix.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

training_job_id="${TRAINING_JOB_ID:?set final exposure-sensitivity array job ID}"
teacher_score_job_id="${TEACHER_SCORE_JOB_ID:?set completed A1 score job ID}"
target_job_id="${TARGET_JOB_ID:?set validated 1k target job ID}"
teacher_score_id="${TEACHER_SCORE_ID:-A1-temporal-k16-seed17000}"
target_name="${TARGET_NAME:-teacher-targets-tomato1k-v1.json}"
for job_id in "${training_job_id}" "${teacher_score_job_id}" "${target_job_id}"; do
  if ! [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "training, teacher-score, and target job IDs must be positive integers" >&2
    exit 2
  fi
done

methods=(B2a-4x B2b-4x B2c-4x B3-4x B4-4x)
evaluation_jobs=()
taste_jobs=()
for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  run_name="${method}-tomato1k-exposure4x-seed17"
  eval_id="${run_name}-temporal-k16"
  model_path="Qwen/Qwen3-4B"
  lora_name="student-adapter"
  lora_path="checkpoints/${run_name}/final"
  generation_config="configs/generation/eval_qwen3_4b_lora.yaml"
  if [[ "${method}" == B4-4x ]]; then
    model_path="checkpoints/${run_name}"
    lora_name=""
    lora_path=""
    generation_config="configs/generation/eval_qwen3_4b.yaml"
  fi
  result="$(
    EVAL_ID="${eval_id}" \
    MODEL_PATH="${model_path}" \
    MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c" \
    SERVED_MODEL_NAME="Qwen/Qwen3-4B" \
    GENERATION_CONFIG="${generation_config}" \
    TRAINING_DEPENDENCY="${training_job_id}_${index}" \
    TEACHER_SCORE_ID="${teacher_score_id}" \
    TEACHER_SCORE_JOB_ID="${teacher_score_job_id}" \
    TARGET_JOB_ID="${target_job_id}" \
    TARGET_NAME="${target_name}" \
    LORA_NAME="${lora_name}" \
    LORA_PATH="${lora_path}" \
    RUN_TASTE="1" \
      scripts/submit_model_evaluation.sh
  )"
  printf '%s\n' "${result}" >&2
  evaluation_job="$(
    printf '%s' "${result}" | uv run python -c \
      'import json,sys; print(json.load(sys.stdin)["evaluation_final"])'
  )"
  taste_job="$(
    printf '%s' "${result}" | uv run python -c \
      'import json,sys; print(json.load(sys.stdin)["taste_final"])'
  )"
  if ! [[ "${evaluation_job}" =~ ^[1-9][0-9]*$ && "${taste_job}" =~ ^[1-9][0-9]*$ ]]; then
    echo "evaluation launcher returned invalid terminal jobs for ${method}" >&2
    exit 1
  fi
  evaluation_jobs+=("${method}:${evaluation_job}")
  taste_jobs+=("${method}:${taste_job}")
done

printf '{"study":"tomato1k-exposure4x","evaluations":"%s","taste":"%s"}\n' \
  "$(IFS=','; printf '%s' "${evaluation_jobs[*]}")" \
  "$(IFS=','; printf '%s' "${taste_jobs[*]}")"
