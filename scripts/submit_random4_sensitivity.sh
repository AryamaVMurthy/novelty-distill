#!/bin/bash
# Submit the unbiased random-4 MRT control and its complete temporal evaluation chain.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

start_after_job_id="${START_AFTER_JOB_ID:?set completed primary-controller dependency}"
teacher_score_job_id="${TEACHER_SCORE_JOB_ID:?set completed A1 score job ID}"
target_job_id="${TARGET_JOB_ID:?set validated 1k target job ID}"
training_passes="${TRAINING_PASSES:-2}"
for job_id in "${start_after_job_id}" "${teacher_score_job_id}" "${target_job_id}"; do
  if ! [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "controller, teacher-score, and target job IDs must be positive integers" >&2
    exit 2
  fi
done
if ! [[ "${training_passes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAINING_PASSES must be a positive integer" >&2
  exit 2
fi

training_job=""
for _pass_index in $(seq 1 "${training_passes}"); do
  if [[ -z "${training_job}" ]]; then
    dependency="afterok:${start_after_job_id}"
  else
    dependency="afterany:${training_job}"
  fi
  submission="$(
    scripts/turing_submit.sh \
      slurm/train_smoke.sbatch \
      --time=06:00:00 \
      --nice=10000 \
      --dependency="${dependency}" \
      --export="ALL,BASELINE_ID=F2-random4,BASELINE_REGISTRY=configs/exposure_sensitivity_baselines.yaml,TRAINING_BACKEND=trl,RUN_CONFIG=configs/training/sft_tomato1k_random4.yaml,RUN_SUFFIX=tomato1k-exposure4x-seed17"
  )"
  training_job="${submission##* }"
  if ! [[ "${training_job}" =~ ^[1-9][0-9]*$ ]]; then
    echo "could not parse random4 training job ID: ${training_job}" >&2
    exit 1
  fi
done

evaluation="$({
  EVAL_ID="F2-random4-tomato1k-exposure4x-seed17-temporal-k16" \
  MODEL_PATH="Qwen/Qwen3-4B" \
  MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c" \
  SERVED_MODEL_NAME="Qwen/Qwen3-4B" \
  GENERATION_CONFIG="configs/generation/eval_qwen3_4b_lora.yaml" \
  TRAINING_DEPENDENCY="${training_job}" \
  TEACHER_SCORE_ID="A1-temporal-k16-seed17000" \
  TEACHER_SCORE_JOB_ID="${teacher_score_job_id}" \
  TARGET_JOB_ID="${target_job_id}" \
  TARGET_NAME="teacher-targets-tomato1k-v1.json" \
  LORA_NAME="student-adapter" \
  LORA_PATH="checkpoints/F2-random4-tomato1k-exposure4x-seed17/final" \
  RUN_TASTE="1" \
    scripts/submit_model_evaluation.sh
})"

TRAINING_JOB_ID="${training_job}" EVALUATION_JSON="${evaluation}" uv run python - <<'PY'
import json
import os

result = json.loads(os.environ["EVALUATION_JSON"])
result["study"] = "tomato1k-random4-mrt"
result["training_final"] = os.environ["TRAINING_JOB_ID"]
print(json.dumps(result, sort_keys=True))
PY
