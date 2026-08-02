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
generation_passes="${GENERATION_PASSES:-3}"
score_passes="${SCORE_PASSES:-2}"
evaluation_passes="${EVALUATION_PASSES:-2}"

for value in "${generation_passes}" "${score_passes}" "${evaluation_passes}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "generation, score, and evaluation passes must be positive integers" >&2
    exit 2
  fi
done

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${REPO_REF:-codex/implementation}"
git -C "${repo_dir}" switch "${REPO_REF:-codex/implementation}"
git -C "${repo_dir}" merge --ff-only "origin/${REPO_REF:-codex/implementation}"
flock -u 9

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

teacher_eval_job=""
for pass_index in $(seq 1 "${evaluation_passes}"); do
  if [[ -z "${teacher_eval_job}" ]]; then
    teacher_eval_dependency="afterok:${teacher_score_job_id}:${target_job_id}"
  else
    teacher_eval_dependency="afterany:${teacher_eval_job}"
  fi
  teacher_eval_job="$(
    submit_job \
      --time=06:00:00 \
      --dependency="${teacher_eval_dependency}" \
      --export="ALL,TEACHER_SCORE_ID=${teacher_score_id},STUDENT_SCORE_ID=${teacher_score_id},SCORE_NAMESPACE=${score_namespace},TARGET_NAME=${target_name},OUTPUT_NAME=A1-temporal-k16-seed17000" \
      slurm/evaluate_student.sbatch
  )"
done
printf '{"baseline":"A1","evaluation":"%s"}\n' "${teacher_eval_job}"

human_prepare_job="$(submit_job slurm/prepare_human_control.sbatch)"
human_score_job=""
for pass_index in $(seq 1 "${score_passes}"); do
  if [[ -z "${human_score_job}" ]]; then
    human_score_dependency="afterok:${human_prepare_job}"
  else
    human_score_dependency="afterany:${human_score_job}"
  fi
  human_score_job="$(
    submit_job \
      --time=06:00:00 \
      --dependency="${human_score_dependency}" \
      --export="ALL,GENERATION_JOB_ID=A3-temporal-k1,GENERATION_NAME=evaluation-student,GENERATION_CONFIG=configs/generation/eval_historical.yaml,PROMPTS_PATH=${input_path},SCORE_NAMESPACE=${score_namespace},SCORE_CONCURRENCY=8" \
      slurm/score_teacher.sbatch
  )"
done
human_eval_job=""
for pass_index in $(seq 1 "${evaluation_passes}"); do
  if [[ -z "${human_eval_job}" ]]; then
    human_eval_dependency="afterok:${teacher_score_job_id}:${human_score_job}:${target_job_id}"
  else
    human_eval_dependency="afterany:${human_eval_job}"
  fi
  human_eval_job="$(
    submit_job \
      --time=06:00:00 \
      --dependency="${human_eval_dependency}" \
      --export="ALL,TEACHER_SCORE_ID=${teacher_score_id},STUDENT_SCORE_ID=A3-temporal-k1,SCORE_NAMESPACE=${score_namespace},TARGET_NAME=${target_name},OUTPUT_NAME=A3-temporal-k1,TEACHER_SAMPLES_PER_PROMPT=16,STUDENT_SAMPLES_PER_PROMPT=1" \
      slurm/evaluate_student.sbatch
  )"
done
printf '{"baseline":"A3","prepare":"%s","score":"%s","evaluation":"%s"}\n' \
  "${human_prepare_job}" "${human_score_job}" "${human_eval_job}"

submit_chain() {
  local baseline_id="$1"
  local model_path="$2"
  local generation_config="$3"
  local lora_path="${4:-}"
  local model_dtype="${5:-auto}"
  local eval_id="${baseline_id}-tomato1k-seed17-temporal-k16"
  local generation_export="ALL,MODEL_PATH=${model_path},MODEL_REVISION=${model_revision},SERVED_MODEL_NAME=Qwen/Qwen3-4B,MODEL_DTYPE=${model_dtype},GENERATION_CONFIG=${generation_config},INPUT_PATH=${input_path},OUTPUT_NAME=evaluation-student,OUTPUT_RUN_ID=${eval_id},INPUT_LIMIT=1658,GENERATION_CONCURRENCY=8"
  if [[ -n "${lora_path}" ]]; then
    generation_export+=",LORA_NAME=student-adapter,LORA_PATH=${lora_path}"
  fi

  local generation_job=""
  local pass_index
  for pass_index in $(seq 1 "${generation_passes}"); do
    local generation_dependency=()
    if [[ -n "${generation_job}" ]]; then
      generation_dependency=(--dependency="afterany:${generation_job}")
    fi
    generation_job="$(
      submit_job \
        --time=06:00:00 \
        "${generation_dependency[@]}" \
        --export="${generation_export}" \
        slurm/sglang_smoke.sbatch
    )"
  done
  local score_job=""
  for pass_index in $(seq 1 "${score_passes}"); do
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
        --export="ALL,GENERATION_JOB_ID=${eval_id},GENERATION_NAME=evaluation-student,GENERATION_CONFIG=${generation_config},PROMPTS_PATH=${input_path},SCORE_NAMESPACE=${score_namespace},SCORE_CONCURRENCY=8" \
        slurm/score_teacher.sbatch
    )"
  done
  local evaluation_job=""
  for pass_index in $(seq 1 "${evaluation_passes}"); do
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
        --export="ALL,TEACHER_SCORE_ID=${teacher_score_id},STUDENT_SCORE_ID=${eval_id},SCORE_NAMESPACE=${score_namespace},TARGET_NAME=${target_name},OUTPUT_NAME=${eval_id}" \
        slurm/evaluate_student.sbatch
    )"
  done
  evaluation_jobs+=("${evaluation_job}")
  printf '{"baseline":"%s","generation_final":"%s","generation_passes":%s,"score_final":"%s","score_passes":%s,"evaluation_final":"%s","evaluation_passes":%s}\n' \
    "${baseline_id}" "${generation_job}" "${generation_passes}" "${score_job}" \
    "${score_passes}" "${evaluation_job}" "${evaluation_passes}"
}

for baseline_id in "${adapter_ids[@]}"; do
  submit_chain \
    "${baseline_id}" \
    Qwen/Qwen3-4B \
    configs/generation/eval_qwen3_4b_lora.yaml \
    "${scratch_root}/checkpoints/${baseline_id}-tomato1k-seed17/final"
done
submit_chain B4 "${gem_path}" configs/generation/eval_qwen3_4b.yaml
submit_chain C3 "${distillm_path}" configs/generation/eval_qwen3_4b.yaml "" bfloat16

evaluation_dependency="$(IFS=:; printf '%s' "${evaluation_jobs[*]}")"
analysis_job="$(
  submit_job \
    --dependency="afterok:${control_eval_job_id}:${teacher_eval_job}:${human_eval_job}:${evaluation_dependency}" \
    slurm/analyze_evaluation_matrix.sbatch
)"
printf '{"analysis":"%s","evaluations":%s}\n' \
  "${analysis_job}" "${#evaluation_jobs[@]}"
