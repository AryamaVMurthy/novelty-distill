#!/bin/bash
# Submit all trained TOMATO-1k temporal-evaluation chains after upstream dependencies succeed.

#SBATCH --job-name=nd-submit-evals
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
teacher_score_job_id="${TEACHER_SCORE_JOB_ID:?set TEACHER_SCORE_JOB_ID}"
target_job_id="${TARGET_JOB_ID:?set TARGET_JOB_ID}"
control_eval_job_id="${CONTROL_EVAL_JOB_ID:-18090}"
teacher_score_id="${TEACHER_SCORE_ID:-A1-temporal-k16-seed17000}"
input_path="${scratch_root}/data/tomato-open-test-1658.jsonl"
model_revision="1cfa9a7208912126459214e8b04321603b3df60c"
score_namespace="evaluation-scores"
target_name="teacher-targets-tomato1k-v1.json"

for job_id in "${teacher_score_job_id}" "${target_job_id}" "${control_eval_job_id}"; do
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "teacher score and target dependencies must be numeric Slurm job IDs" >&2
    exit 2
  fi
done
test -f "${input_path}"
test -f "${scratch_root}/data/${target_name}"
test -d "${scratch_root}/${score_namespace}/${teacher_score_id}"

adapter_ids=(
  B1 B2a B2b B2c B3
  C1-human C1-best1 C1-diverse4 C2-human C2-best1 C2-diverse4
  D1 D2 D3 E2 E3 E4
)
evaluation_jobs=()
for baseline_id in "${adapter_ids[@]}"; do
  adapter_path="${scratch_root}/checkpoints/${baseline_id}-tomato1k-seed17/final"
  test -f "${adapter_path}/adapter_config.json"
  test -f "${adapter_path}/adapter_model.safetensors"
done
gem_path="${scratch_root}/checkpoints/B4-tomato1k-seed17"
distillm_path="${scratch_root}/checkpoints/C3-tomato1k-seed17-l896-4gpu/250"
for full_path in "${gem_path}" "${distillm_path}"; do
  test -f "${full_path}/config.json"
  find "${full_path}" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name '*.index.json' \) \
    -print -quit | grep -q .
done

cd "${repo_dir}"

submit_job() {
  local job_id
  job_id="$(
    sbatch --parsable \
      --output="${scratch_root}/logs/slurm-%x-%j.out" \
      --error="${scratch_root}/logs/slurm-%x-%j.err" \
      "$@"
  )"
  job_id="${job_id%%;*}"
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "could not parse submitted job ID: ${job_id}" >&2
    exit 1
  fi
  printf '%s\n' "${job_id}"
}

submit_chain() {
  local baseline_id="$1"
  local model_path="$2"
  local generation_config="$3"
  local lora_path="${4:-}"
  local eval_id="${baseline_id}-tomato1k-seed17-temporal-k16"
  local generation_export="ALL,MODEL_PATH=${model_path},MODEL_REVISION=${model_revision},SERVED_MODEL_NAME=Qwen/Qwen3-4B,GENERATION_CONFIG=${generation_config},INPUT_PATH=${input_path},OUTPUT_NAME=evaluation-student,OUTPUT_RUN_ID=${eval_id},INPUT_LIMIT=1658,GENERATION_CONCURRENCY=8"
  if [[ -n "${lora_path}" ]]; then
    generation_export+=",LORA_NAME=student-adapter,LORA_PATH=${lora_path}"
  fi

  local generation_job
  generation_job="$(
    submit_job --time=06:00:00 --export="${generation_export}" slurm/sglang_smoke.sbatch
  )"
  local score_job
  score_job="$(
    submit_job \
      --time=06:00:00 \
      --dependency="afterok:${generation_job}" \
      --export="ALL,GENERATION_JOB_ID=${eval_id},GENERATION_NAME=evaluation-student,GENERATION_CONFIG=${generation_config},PROMPTS_PATH=${input_path},SCORE_NAMESPACE=${score_namespace},SCORE_CONCURRENCY=8" \
      slurm/score_teacher.sbatch
  )"
  local evaluation_job
  evaluation_job="$(
    submit_job \
      --time=06:00:00 \
      --dependency="afterok:${teacher_score_job_id}:${score_job}:${target_job_id}" \
      --export="ALL,TEACHER_SCORE_ID=${teacher_score_id},STUDENT_SCORE_ID=${eval_id},SCORE_NAMESPACE=${score_namespace},TARGET_NAME=${target_name},OUTPUT_NAME=${eval_id}" \
      slurm/evaluate_student.sbatch
  )"
  evaluation_jobs+=("${evaluation_job}")
  printf '{"baseline":"%s","generation":"%s","score":"%s","evaluation":"%s"}\n' \
    "${baseline_id}" "${generation_job}" "${score_job}" "${evaluation_job}"
}

for baseline_id in "${adapter_ids[@]}"; do
  submit_chain \
    "${baseline_id}" \
    Qwen/Qwen3-4B \
    configs/generation/eval_qwen3_4b_lora.yaml \
    "${scratch_root}/checkpoints/${baseline_id}-tomato1k-seed17/final"
done
submit_chain B4 "${gem_path}" configs/generation/eval_qwen3_4b.yaml
submit_chain C3 "${distillm_path}" configs/generation/eval_qwen3_4b.yaml

evaluation_dependency="$(IFS=:; printf '%s' "${evaluation_jobs[*]}")"
analysis_job="$(
  submit_job \
    --dependency="afterok:${control_eval_job_id}:${evaluation_dependency}" \
    slurm/analyze_evaluation_matrix.sbatch
)"
printf '{"analysis":"%s","evaluations":%s}\n' \
  "${analysis_job}" "${#evaluation_jobs[@]}"
