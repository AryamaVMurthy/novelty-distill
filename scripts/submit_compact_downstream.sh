#!/bin/bash
# Resume compact K=4 scoring/evaluation from existing generation arrays.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

a0_evaluation_job_id="${A0_EVALUATION_JOB_ID:?set A0_EVALUATION_JOB_ID}"
generation_job_ids_csv="${GENERATION_JOB_IDS:?set GENERATION_JOB_IDS in baseline order}"
IFS=, read -r -a generation_job_ids <<<"${generation_job_ids_csv}"
baseline_ids=(B1 B2b C1-best1 C2-best1 D1)
artifact_identities=(
  sha256:cfb6e32ef2a0648b6d8e372f0d2325098faa352c8d3a6f4af1cfc55fa75cf280
  sha256:ee15f4037d9b0e92b79cb5b6291616f9d618a925d597e91938d7387e0ac6f2a8
  sha256:7fc74fcd56d0150faf987eb5234536cac649cd24882a4e35716ff2aedc1b50f8
  sha256:7080e929e3f72d2c58c4ef01b6afc86f8c190b351d79c8039fecff01eb6d9cf4
  sha256:47a46794c06fdfdbe9b9109171de22c7588c00b5eb67d2e2ee2ed2f4d9ddd216
)
student_config="configs/generation/eval_qwen3_4b_lora_k4.yaml"
input_name="tomato-open-test-1658.jsonl"
teacher_id="A1-temporal-k16-seed17000"
expected_prompts=1658

if [[ ! "${a0_evaluation_job_id}" =~ ^[0-9]+$ ]]; then
  echo "A0_EVALUATION_JOB_ID must be numeric" >&2
  exit 2
fi
if ((${#generation_job_ids[@]} != ${#baseline_ids[@]})); then
  echo "GENERATION_JOB_IDS must contain five comma-separated job IDs" >&2
  exit 2
fi
for job_id in "${generation_job_ids[@]}"; do
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "every generation job ID must be numeric" >&2
    exit 2
  fi
done

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

evaluation_jobs=()
score_arrays=()
for index in "${!baseline_ids[@]}"; do
  baseline_id="${baseline_ids[index]}"
  generation_job_id="${generation_job_ids[index]}"
  artifact_identity="${artifact_identities[index]}"
  eval_id="${baseline_id}-tomato1k-seed17-temporal-k4"
  generation_gate="$(
    submit_job slurm/validate_generation_run.sbatch \
      --dependency="afterok:${generation_job_id}" \
      --export="ALL,GENERATION_NAME=evaluation-student,GENERATION_ID=${eval_id},GENERATION_CONFIG=${student_config},INPUT_NAME=${input_name},EXPECTED_PROMPTS=${expected_prompts},SERVED_ARTIFACT_IDENTITY=${artifact_identity}"
  )"
  score_array="$(
    submit_job slurm/score_teacher.sbatch \
      --array="0-3%4" \
      --time=03:00:00 \
      --dependency="afterok:${generation_gate}" \
      --export="ALL,GENERATION_JOB_ID=${eval_id},GENERATION_NAME=evaluation-student,GENERATION_CONFIG=${student_config},PROMPTS_NAME=${input_name},SCORE_NAMESPACE=evaluation-scores,SCORE_CONCURRENCY=8,SCORE_NUM_SHARDS=4"
  )"
  score_arrays+=("${score_array}")
  score_gate="$(
    submit_job slurm/validate_score_run.sbatch \
      --dependency="afterok:${score_array}" \
      --export="ALL,GENERATION_NAME=evaluation-student,GENERATION_ID=${eval_id},GENERATION_CONFIG=${student_config},SCORE_NAMESPACE=evaluation-scores,EXPECTED_PROMPTS=${expected_prompts}"
  )"
  evaluation_job="$(
    submit_job slurm/evaluate_student.sbatch \
      --time=03:00:00 \
      --dependency="afterok:${a0_evaluation_job_id}:${score_gate}" \
      --export="ALL,TEACHER_SCORE_ID=${teacher_id},STUDENT_SCORE_ID=${eval_id},SCORE_NAMESPACE=evaluation-scores,TARGET_NAME=teacher-targets-tomato1k-v1.json,OUTPUT_NAME=${eval_id},TEACHER_SAMPLES_PER_PROMPT=4,STUDENT_SAMPLES_PER_PROMPT=4,TEACHER_SOURCE_SAMPLES_PER_PROMPT=16,STUDENT_SOURCE_SAMPLES_PER_PROMPT=4"
  )"
  evaluation_jobs+=("${evaluation_job}")
done

analysis_dependency="$(IFS=:; printf '%s' "${evaluation_jobs[*]}")"
analysis_job="$(
  submit_job slurm/analyze_compact_baselines.sbatch \
    --dependency="afterok:${analysis_dependency}"
)"

printf '{"study":"tomato1k-compact-k4","generation_jobs":"%s","score_arrays":"%s","evaluation_jobs":"%s","analysis_job":"%s"}\n' \
  "${generation_job_ids[*]}" "${score_arrays[*]}" "${evaluation_jobs[*]}" "${analysis_job}"
