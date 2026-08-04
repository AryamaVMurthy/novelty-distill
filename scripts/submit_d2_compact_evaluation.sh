#!/bin/bash
# Submit the complete D2 K=4 generation, judging, evaluation, and extension analysis chain.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

eval_id="D2-tomato1k-seed17-temporal-k4"
student_config="configs/generation/eval_qwen3_4b_lora_k4.yaml"
input_name="tomato-open-test-1658.jsonl"
teacher_id="A1-temporal-k16-seed17000"
expected_prompts=1658
adapter_path="checkpoints/D2-tomato1k-seed17/final"
artifact_identity="sha256:56904eba6ecb94626bd6d9cb10f1afda48ce5a7571b6f3a6181f4db662359c46"
target_node="${TARGET_NODE:-node01}"
if [[ ! "${target_node}" =~ ^node[0-9]{2}$ ]]; then
  echo "TARGET_NODE must have the form nodeNN" >&2
  exit 2
fi

submit_job() {
  local output job_id
  output="$(scripts/turing_submit.sh "$@")"
  printf '%s\n' "${output}" >&2
  job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job/{print $4}' | tail -1)"
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "could not parse submitted job ID" >&2
    exit 1
  fi
  printf '%s\n' "${job_id}"
}

generation_array="$(
  submit_job slurm/sglang_smoke.sbatch \
    --nodelist="${target_node}" \
    --array="0-3%4" \
    --time=03:00:00 \
    --export="ALL,MODEL_PATH=Qwen/Qwen3-4B,MODEL_REVISION=1cfa9a7208912126459214e8b04321603b3df60c,SERVED_MODEL_NAME=Qwen/Qwen3-4B,GENERATION_CONFIG=${student_config},INPUT_NAME=${input_name},OUTPUT_NAME=evaluation-student,OUTPUT_RUN_ID=${eval_id},INPUT_LIMIT=${expected_prompts},GENERATION_CONCURRENCY=8,GENERATION_NUM_SHARDS=4,LORA_NAME=student-adapter,LORA_PATH=${adapter_path}"
)"
generation_gate="$(
  submit_job slurm/validate_generation_run.sbatch \
    --nodelist="${target_node}" \
    --dependency="afterok:${generation_array}" \
    --export="ALL,GENERATION_NAME=evaluation-student,GENERATION_ID=${eval_id},GENERATION_CONFIG=${student_config},INPUT_NAME=${input_name},EXPECTED_PROMPTS=${expected_prompts},SERVED_ARTIFACT_IDENTITY=${artifact_identity}"
)"
score_array="$(
  submit_job slurm/score_teacher.sbatch \
    --nodelist="${target_node}" \
    --array="0-3%4" \
    --time=03:00:00 \
    --dependency="afterok:${generation_gate}" \
    --export="ALL,GENERATION_JOB_ID=${eval_id},GENERATION_NAME=evaluation-student,GENERATION_CONFIG=${student_config},PROMPTS_NAME=${input_name},SCORE_NAMESPACE=evaluation-scores,SCORE_CONCURRENCY=8,SCORE_NUM_SHARDS=4"
)"
score_gate="$(
  submit_job slurm/validate_score_run.sbatch \
    --nodelist="${target_node}" \
    --dependency="afterok:${score_array}" \
    --export="ALL,GENERATION_NAME=evaluation-student,GENERATION_ID=${eval_id},GENERATION_CONFIG=${student_config},SCORE_NAMESPACE=evaluation-scores,EXPECTED_PROMPTS=${expected_prompts}"
)"
evaluation_job="$(
  submit_job slurm/evaluate_student.sbatch \
    --nodelist="${target_node}" \
    --time=03:00:00 \
    --dependency="afterok:${score_gate}" \
    --export="ALL,TEACHER_SCORE_ID=${teacher_id},STUDENT_SCORE_ID=${eval_id},SCORE_NAMESPACE=evaluation-scores,TARGET_NAME=teacher-targets-tomato1k-v1.json,OUTPUT_NAME=${eval_id},TEACHER_SAMPLES_PER_PROMPT=4,STUDENT_SAMPLES_PER_PROMPT=4,TEACHER_SOURCE_SAMPLES_PER_PROMPT=16,STUDENT_SOURCE_SAMPLES_PER_PROMPT=4"
)"
analysis_job="$(
  submit_job slurm/analyze_d2_compact_extension.sbatch \
    --nodelist="${target_node}" \
    --dependency="afterok:${evaluation_job}"
)"

printf '{"study":"tomato1k-compact-k4-d2-extension","target_node":"%s","generation_array":"%s","generation_gate":"%s","score_array":"%s","score_gate":"%s","evaluation_job":"%s","analysis_job":"%s"}\n' \
  "${target_node}" "${generation_array}" "${generation_gate}" "${score_array}" "${score_gate}" \
  "${evaluation_job}" "${analysis_job}"
