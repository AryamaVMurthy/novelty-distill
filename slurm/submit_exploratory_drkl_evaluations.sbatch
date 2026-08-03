#!/bin/bash
# Submit secondary DRKL temporal/taste chains and a matched post-primary analysis.

#SBATCH --job-name=nd-submit-drkl
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
best1_training_job_id="${BEST1_TRAINING_JOB_ID:?set BEST1_TRAINING_JOB_ID}"
diverse4_training_job_id="${DIVERSE4_TRAINING_JOB_ID:?set DIVERSE4_TRAINING_JOB_ID}"
primary_controller_job_id="${PRIMARY_CONTROLLER_JOB_ID:-18222}"
teacher_score_job_id="${TEACHER_SCORE_JOB_ID:-18202}"
target_job_id="${TARGET_JOB_ID:-18206}"
input_path="${scratch_root}/data/tomato-open-test-1658.jsonl"
generation_config="configs/generation/eval_qwen3_4b_lora.yaml"
model_revision="1cfa9a7208912126459214e8b04321603b3df60c"

for job_id in \
  "${best1_training_job_id}" "${diverse4_training_job_id}" \
  "${primary_controller_job_id}" "${teacher_score_job_id}" "${target_job_id}"
do
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "all dependency IDs must be numeric" >&2
    exit 2
  fi
done

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" \
  "${repo_dir}" "${REPO_REF:-codex/implementation}"
flock -u 9

test -f "${input_path}"
for baseline_id in F1-best1 F1-diverse4; do
  adapter_path="${scratch_root}/checkpoints/${baseline_id}-exploratory-drkl-tomato1k-seed17/final"
  test -f "${adapter_path}/adapter_config.json"
  test -f "${adapter_path}/adapter_model.safetensors"
done

primary_log="${scratch_root}/logs/slurm-nd-submit-evals-${primary_controller_job_id}.out"
test -s "${primary_log}"
primary_analysis_job_id="$(
  grep -oE '"analysis":"[0-9]+"' "${primary_log}" | tail -1 | tr -cd '0-9'
)"
if ! [[ "${primary_analysis_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not recover primary analysis job ID from ${primary_log}" >&2
  exit 1
fi

cd "${repo_dir}"
submit_job() {
  local job_id
  job_id="$(
    sbatch --parsable \
      --nice=10000 \
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

evaluation_jobs=()
taste_jobs=()
submit_chain() {
  local baseline_id="$1"
  local training_job_id="$2"
  local eval_id="${baseline_id}-exploratory-drkl-tomato1k-seed17-temporal-k16"
  local adapter_path="${scratch_root}/checkpoints/${baseline_id}-exploratory-drkl-tomato1k-seed17/final"
  local generation_export="ALL,MODEL_PATH=Qwen/Qwen3-4B,MODEL_REVISION=${model_revision},SERVED_MODEL_NAME=Qwen/Qwen3-4B,MODEL_DTYPE=auto,GENERATION_CONFIG=${generation_config},INPUT_PATH=${input_path},OUTPUT_NAME=evaluation-student,OUTPUT_RUN_ID=${eval_id},INPUT_LIMIT=1658,GENERATION_CONCURRENCY=8,LORA_NAME=student-adapter,LORA_PATH=${adapter_path}"

  local generation_job=""
  local pass
  for pass in 1 2 3; do
    local generation_dependency
    if [[ -z "${generation_job}" ]]; then
      generation_dependency="afterok:${training_job_id}"
    else
      generation_dependency="afterany:${generation_job}"
    fi
    generation_job="$(
      submit_job \
        --time=06:00:00 \
        --dependency="${generation_dependency}" \
        --export="${generation_export}" \
        slurm/sglang_smoke.sbatch
    )"
  done

  local score_job=""
  for pass in 1 2; do
    local score_dependency
    if [[ -z "${score_job}" ]]; then
      score_dependency="afterok:${generation_job}"
    else
      score_dependency="afterany:${score_job}"
    fi
    score_job="$(
      submit_job \
        --time=06:00:00 \
        --dependency="${score_dependency}" \
        --export="ALL,GENERATION_JOB_ID=${eval_id},GENERATION_NAME=evaluation-student,GENERATION_CONFIG=${generation_config},PROMPTS_PATH=${input_path},SCORE_NAMESPACE=evaluation-scores,SCORE_CONCURRENCY=8" \
        slurm/score_teacher.sbatch
    )"
  done

  local evaluation_job=""
  for pass in 1 2; do
    local evaluation_dependency
    if [[ -z "${evaluation_job}" ]]; then
      evaluation_dependency="afterok:${teacher_score_job_id}:${score_job}:${target_job_id}"
    else
      evaluation_dependency="afterany:${evaluation_job}"
    fi
    evaluation_job="$(
      submit_job \
        --time=06:00:00 \
        --dependency="${evaluation_dependency}" \
        --export="ALL,TEACHER_SCORE_ID=A1-temporal-k16-seed17000,STUDENT_SCORE_ID=${eval_id},SCORE_NAMESPACE=evaluation-scores,TARGET_NAME=teacher-targets-tomato1k-v1.json,OUTPUT_NAME=${eval_id}" \
        slurm/evaluate_student.sbatch
    )"
  done

  local taste_job=""
  for pass in 1 2; do
    local taste_dependency
    if [[ -z "${taste_job}" ]]; then
      taste_dependency="afterok:${generation_job}"
    else
      taste_dependency="afterany:${taste_job}"
    fi
    taste_job="$(
      submit_job \
        --time=06:00:00 \
        --dependency="${taste_dependency}" \
        --export="ALL,GENERATION_JOB_ID=${eval_id},GENERATION_NAME=evaluation-student,GENERATION_CONFIG=${generation_config},PROMPTS_PATH=${input_path},TASTE_NAMESPACE=research-taste,TASTE_CONCURRENCY=8" \
        slurm/annotate_research_taste.sbatch
    )"
  done

  evaluation_jobs+=("${evaluation_job}")
  taste_jobs+=("${taste_job}")
  printf '{"baseline":"%s","generation":"%s","score":"%s","evaluation":"%s","taste":"%s"}\n' \
    "${baseline_id}" "${generation_job}" "${score_job}" "${evaluation_job}" "${taste_job}"
}

submit_chain F1-best1 "${best1_training_job_id}"
submit_chain F1-diverse4 "${diverse4_training_job_id}"

evaluation_dependency="$(IFS=:; printf '%s' "${evaluation_jobs[*]}")"
taste_dependency="$(IFS=:; printf '%s' "${taste_jobs[*]}")"
analysis_job="$(
  submit_job \
    --dependency="afterok:${primary_analysis_job_id}:${evaluation_dependency}:${taste_dependency}" \
    slurm/analyze_exploratory_drkl.sbatch
)"
printf '{"status":"secondary_exploratory","primary_analysis":"%s","analysis":"%s"}\n' \
  "${primary_analysis_job_id}" "${analysis_job}"
