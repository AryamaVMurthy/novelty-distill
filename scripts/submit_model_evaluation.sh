#!/bin/bash
# Submit one resumable generation -> judge -> joint-evaluation chain on TOMATO temporal test.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

eval_id="${EVAL_ID:?set EVAL_ID}"
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"
model_revision="${MODEL_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"
served_model_name="${SERVED_MODEL_NAME:-Qwen/Qwen3-4B}"
generation_config="${GENERATION_CONFIG:-configs/generation/eval_qwen3_4b.yaml}"
generation_name="${GENERATION_NAME:-evaluation-student}"
input_path="${INPUT_PATH:-/scratch/aryama.murthy/novelty-distill/data/tomato-open-test-1658.jsonl}"
input_limit="${INPUT_LIMIT:-1658}"
teacher_score_id="${TEACHER_SCORE_ID:-A1-temporal-k16-seed17000}"
teacher_score_job_id="${TEACHER_SCORE_JOB_ID:?set TEACHER_SCORE_JOB_ID}"
target_job_id="${TARGET_JOB_ID:?set TARGET_JOB_ID}"
target_name="${TARGET_NAME:-teacher-targets-tomato1k-v1.json}"
training_dependency="${TRAINING_DEPENDENCY:-}"
lora_name="${LORA_NAME:-}"
lora_path="${LORA_PATH:-}"
generation_passes="${GENERATION_PASSES:-3}"
score_passes="${SCORE_PASSES:-2}"

for value in "${eval_id}" "${generation_name}" "${teacher_score_id}" "${target_name}"; do
  if [[ -z "${value}" || "${value}" == */* || "${value}" == *..* || "${value}" == *,* ]]; then
    echo "evaluation IDs and names must be path-safe and comma-free" >&2
    exit 2
  fi
done
for value in "${model_path}" "${model_revision}" "${served_model_name}" "${generation_config}" "${input_path}"; do
  if [[ -z "${value}" || "${value}" == *,* ]]; then
    echo "exported evaluation values must be non-empty and comma-free" >&2
    exit 2
  fi
done
for job_id in "${teacher_score_job_id}" "${target_job_id}"; do
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "teacher score and target dependencies must be numeric Slurm job IDs" >&2
    exit 2
  fi
done
if [[ -n "${training_dependency}" ]] && ! [[ "${training_dependency}" =~ ^[0-9]+$ ]]; then
  echo "TRAINING_DEPENDENCY must be a numeric Slurm job ID" >&2
  exit 2
fi
if ! [[ "${input_limit}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INPUT_LIMIT must be a positive integer" >&2
  exit 2
fi
for value in "${generation_passes}" "${score_passes}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GENERATION_PASSES and SCORE_PASSES must be positive integers" >&2
    exit 2
  fi
done
if [[ -n "${lora_name}" || -n "${lora_path}" ]]; then
  if [[ -z "${lora_name}" || -z "${lora_path}" || "${lora_name}" == *,* || "${lora_path}" == *,* ]]; then
    echo "LORA_NAME and LORA_PATH must be set together and comma-free" >&2
    exit 2
  fi
fi

submit_job() {
  local output
  output="$(scripts/turing_submit.sh "$@")"
  printf '%s\n' "${output}" >&2
  local job_id
  job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job/{print $4}' | tail -1)"
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "could not parse submitted job ID" >&2
    exit 1
  fi
  printf '%s\n' "${job_id}"
}

generation_export="ALL,MODEL_PATH=${model_path},MODEL_REVISION=${model_revision},SERVED_MODEL_NAME=${served_model_name},GENERATION_CONFIG=${generation_config},INPUT_PATH=${input_path},OUTPUT_NAME=${generation_name},OUTPUT_RUN_ID=${eval_id},INPUT_LIMIT=${input_limit},GENERATION_CONCURRENCY=8"
if [[ -n "${lora_name}" ]]; then
  generation_export+=",LORA_NAME=${lora_name},LORA_PATH=${lora_path}"
fi
generation_job=""
for ((pass = 1; pass <= generation_passes; pass++)); do
  generation_dependency=()
  if [[ -n "${generation_job}" ]]; then
    generation_dependency=(--dependency="afterany:${generation_job}")
  elif [[ -n "${training_dependency}" ]]; then
    generation_dependency=(--dependency="afterok:${training_dependency}")
  fi
  generation_job="$(
    submit_job slurm/sglang_smoke.sbatch \
      --time=06:00:00 \
      "${generation_dependency[@]}" \
      --export="${generation_export}"
  )"
done

score_job=""
for ((pass = 1; pass <= score_passes; pass++)); do
  if [[ -z "${score_job}" ]]; then
    score_dependency="afterok:${generation_job}"
  else
    score_dependency="afterany:${score_job}"
  fi
  score_job="$(
    submit_job slurm/score_teacher.sbatch \
      --time=06:00:00 \
      --dependency="${score_dependency}" \
      --export="ALL,GENERATION_JOB_ID=${eval_id},GENERATION_NAME=${generation_name},GENERATION_CONFIG=${generation_config},PROMPTS_PATH=${input_path},SCORE_NAMESPACE=evaluation-scores,SCORE_CONCURRENCY=8"
  )"
done

evaluation_job="$(
  submit_job slurm/evaluate_student.sbatch \
    --time=06:00:00 \
    --dependency="afterok:${teacher_score_job_id}:${score_job}:${target_job_id}" \
    --export="ALL,TEACHER_SCORE_ID=${teacher_score_id},STUDENT_SCORE_ID=${eval_id},SCORE_NAMESPACE=evaluation-scores,TARGET_NAME=${target_name},OUTPUT_NAME=${eval_id}"
)"

printf '{"evaluation_id":"%s","generation_final":"%s","generation_passes":%s,"score_final":"%s","score_passes":%s,"evaluation":"%s"}\n' \
  "${eval_id}" "${generation_job}" "${generation_passes}" "${score_job}" \
  "${score_passes}" "${evaluation_job}"
