#!/bin/bash
# Submit independent Qwen3-8B teacher and untouched Qwen3-1.7B temporal controls.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

target_job_id="${TARGET_JOB_ID:?set replication 1k teacher-target gate job ID}"
target_name="${TARGET_NAME:-teacher-targets-qwen3-8b-tomato1000-v1.json}"
generation_passes="${GENERATION_PASSES:-3}"
score_passes="${SCORE_PASSES:-2}"
taste_passes="${TASTE_PASSES:-2}"
evaluation_passes="${EVALUATION_PASSES:-2}"
teacher_id="A1-qwen8b-temporal-k16-seed17000"
student_id="A0-qwen1p7b-temporal-k16-seed17000"
prompt_name="tomato-open-test-1658.jsonl"
ids=("${teacher_id}" "${student_id}")
models=("Qwen/Qwen3-8B" "Qwen/Qwen3-1.7B")
revisions=(
  "b968826d9c46dd6066d109eabc6255188de91218"
  "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
configs=(
  "configs/generation/eval_qwen3_8b.yaml"
  "configs/generation/eval_qwen3_1p7b.yaml"
)
names=("evaluation-teacher" "evaluation-student")
for value in "${target_job_id}" "${generation_passes}" "${score_passes}" \
  "${taste_passes}" "${evaluation_passes}"
do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "job IDs and pass counts must be positive integers" >&2
    exit 2
  fi
done
if [[ -z "${target_name}" || "${target_name}" == */* || "${target_name}" == *..* ]]; then
  echo "TARGET_NAME must be path-safe" >&2
  exit 2
fi

submit_job() {
  local output job_id
  output="$(scripts/turing_submit.sh "$@")"
  printf '%s\n' "${output}" >&2
  job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job/{print $4}' | tail -1)"
  if ! [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "could not parse submitted job ID" >&2
    exit 1
  fi
  printf '%s\n' "${job_id}"
}

generation_finals=()
score_finals=()
taste_finals=()
for index in 0 1; do
  generation_job=""
  generation_export="ALL,MODEL_PATH=${models[$index]},MODEL_REVISION=${revisions[$index]},SERVED_MODEL_NAME=${models[$index]},GENERATION_CONFIG=${configs[$index]},INPUT_NAME=${prompt_name},OUTPUT_NAME=${names[$index]},OUTPUT_RUN_ID=${ids[$index]},INPUT_LIMIT=1658,GENERATION_CONCURRENCY=8"
  for ((pass = 1; pass <= generation_passes; pass++)); do
    if [[ -z "${generation_job}" ]]; then
      dependency="afterok:${target_job_id}"
    else
      dependency="afterany:${generation_job}"
    fi
    generation_job="$(
      submit_job slurm/sglang_smoke.sbatch \
        --time=06:00:00 \
        --dependency="${dependency}" \
        --export="${generation_export}"
    )"
  done
  generation_finals+=("${generation_job}")

  score_job=""
  for ((pass = 1; pass <= score_passes; pass++)); do
    if [[ -z "${score_job}" ]]; then
      dependency="afterok:${generation_job}"
    else
      dependency="afterany:${score_job}"
    fi
    score_job="$(
      submit_job slurm/score_teacher.sbatch \
        --time=06:00:00 \
        --dependency="${dependency}" \
        --export="ALL,GENERATION_JOB_ID=${ids[$index]},GENERATION_NAME=${names[$index]},GENERATION_CONFIG=${configs[$index]},PROMPTS_NAME=${prompt_name},SCORE_NAMESPACE=evaluation-scores,SCORE_CONCURRENCY=8"
    )"
  done
  score_finals+=("${score_job}")

  taste_job=""
  for ((pass = 1; pass <= taste_passes; pass++)); do
    if [[ -z "${taste_job}" ]]; then
      dependency="afterok:${generation_job}"
    else
      dependency="afterany:${taste_job}"
    fi
    taste_job="$(
      submit_job slurm/annotate_research_taste.sbatch \
        --time=06:00:00 \
        --dependency="${dependency}" \
        --export="ALL,GENERATION_JOB_ID=${ids[$index]},GENERATION_NAME=${names[$index]},GENERATION_CONFIG=${configs[$index]},PROMPTS_NAME=${prompt_name},TASTE_NAMESPACE=research-taste,TASTE_CONCURRENCY=8"
    )"
  done
  taste_finals+=("${taste_job}")
done

teacher_eval=""
for ((pass = 1; pass <= evaluation_passes; pass++)); do
  if [[ -z "${teacher_eval}" ]]; then
    dependency="afterok:${score_finals[0]}:${target_job_id}"
  else
    dependency="afterany:${teacher_eval}"
  fi
  teacher_eval="$(
    submit_job slurm/evaluate_student.sbatch \
      --dependency="${dependency}" \
      --export="ALL,TEACHER_SCORE_ID=${teacher_id},STUDENT_SCORE_ID=${teacher_id},SCORE_NAMESPACE=evaluation-scores,TARGET_NAME=${target_name},OUTPUT_NAME=${teacher_id}"
  )"
done

student_eval=""
for ((pass = 1; pass <= evaluation_passes; pass++)); do
  if [[ -z "${student_eval}" ]]; then
    dependency="afterok:${score_finals[0]}:${score_finals[1]}:${target_job_id}"
  else
    dependency="afterany:${student_eval}"
  fi
  student_eval="$(
    submit_job slurm/evaluate_student.sbatch \
      --dependency="${dependency}" \
      --export="ALL,TEACHER_SCORE_ID=${teacher_id},STUDENT_SCORE_ID=${student_id},SCORE_NAMESPACE=evaluation-scores,TARGET_NAME=${target_name},OUTPUT_NAME=${student_id}"
  )"
done

printf '{"teacher_id":"%s","student_id":"%s","teacher_generation_final":"%s","student_generation_final":"%s","teacher_score_final":"%s","student_score_final":"%s","teacher_taste_final":"%s","student_taste_final":"%s","teacher_evaluation_final":"%s","student_evaluation_final":"%s"}\n' \
  "${teacher_id}" "${student_id}" "${generation_finals[0]}" "${generation_finals[1]}" \
  "${score_finals[0]}" "${score_finals[1]}" "${taste_finals[0]}" "${taste_finals[1]}" \
  "${teacher_eval}" "${student_eval}"
