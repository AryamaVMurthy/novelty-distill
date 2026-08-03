#!/bin/bash
# Submit a fail-closed four-GPU main or replication teacher bank.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

train_size="${TRAIN_SIZE:-5000}"
teacher_profile="${TEACHER_PROFILE:-main}"
generation_passes="${GENERATION_PASSES:-2}"
score_passes="${SCORE_PASSES:-2}"
num_gpu_shards="${NUM_GPU_SHARDS:-4}"
start_after_job_id="${START_AFTER_JOB_ID:-}"
case "${teacher_profile}" in
  main)
    case "${train_size}" in
      5000|20000) ;;
      *) echo "main TRAIN_SIZE must be 5000 or 20000" >&2; exit 2 ;;
    esac
    generation_config="configs/generation/teacher.yaml"
    teacher_model="Qwen/Qwen3-14B"
    teacher_revision="40c069824f4251a91eefaf281ebe4c544efd3e18"
    generation_id="teacher-${train_size}-v1"
    target_name="teacher-targets-tomato${train_size}-v1.json"
    if [[ "${train_size}" == 5000 ]]; then
      source_generation_id="teacher-1k-v1"
    else
      source_generation_id="teacher-5000-v1"
    fi
    ;;
  replication)
    case "${train_size}" in
      1000|5000|20000) ;;
      *) echo "replication TRAIN_SIZE must be 1000, 5000, or 20000" >&2; exit 2 ;;
    esac
    generation_config="configs/generation/teacher_qwen3_8b.yaml"
    teacher_model="Qwen/Qwen3-8B"
    teacher_revision="b968826d9c46dd6066d109eabc6255188de91218"
    generation_id="teacher8b-${train_size}-v1"
    target_name="teacher-targets-qwen3-8b-tomato${train_size}-v1.json"
    if [[ "${train_size}" == 1000 ]]; then
      source_generation_id=""
    elif [[ "${train_size}" == 5000 ]]; then
      source_generation_id="teacher8b-1000-v1"
    else
      source_generation_id="teacher8b-5000-v1"
    fi
    ;;
  *) echo "TEACHER_PROFILE must be main or replication" >&2; exit 2 ;;
esac
for value in "${generation_passes}" "${score_passes}" "${num_gpu_shards}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "pass and GPU shard counts must be positive integers" >&2
    exit 2
  fi
done
if [[ "${num_gpu_shards}" -ne 4 ]]; then
  echo "NUM_GPU_SHARDS must be 4 for the production scale protocol" >&2
  exit 2
fi
if [[ -n "${start_after_job_id}" && ! "${start_after_job_id}" =~ ^[0-9]+$ ]]; then
  echo "START_AFTER_JOB_ID must be a numeric Slurm job ID" >&2
  exit 2
fi

input_name="tomato-open-train-${train_size}.jsonl"

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

bootstrap_job=""
initial_dependency=()
if [[ -n "${source_generation_id}" ]]; then
  bootstrap_dependency=()
  if [[ -n "${start_after_job_id}" ]]; then
    bootstrap_dependency=(--dependency="afterok:${start_after_job_id}")
  fi
  bootstrap_job="$(
    submit_job slurm/bootstrap_generation_run.sbatch \
      "${bootstrap_dependency[@]}" \
      --export="ALL,SOURCE_GENERATION_ID=${source_generation_id},TARGET_GENERATION_ID=${generation_id},INPUT_NAME=${input_name},GENERATION_CONFIG=${generation_config}"
  )"
  initial_dependency=(--dependency="afterok:${bootstrap_job}")
elif [[ -n "${start_after_job_id}" ]]; then
  initial_dependency=(--dependency="afterok:${start_after_job_id}")
fi

generation_job=""
for ((pass = 1; pass <= generation_passes; pass++)); do
  if [[ -z "${generation_job}" ]]; then
    dependency=("${initial_dependency[@]}")
  else
    dependency=(--dependency="afterany:${generation_job}")
  fi
  generation_job="$(
    submit_job slurm/sglang_smoke.sbatch \
      --array="0-$((num_gpu_shards - 1))%${num_gpu_shards}" \
      --time=12:00:00 \
      "${dependency[@]}" \
      --export="ALL,MODEL_PATH=${teacher_model},MODEL_REVISION=${teacher_revision},SERVED_MODEL_NAME=${teacher_model},GENERATION_CONFIG=${generation_config},OUTPUT_NAME=teacher,OUTPUT_RUN_ID=${generation_id},INPUT_NAME=${input_name},INPUT_LIMIT=${train_size},GENERATION_CONCURRENCY=8,GENERATION_NUM_SHARDS=${num_gpu_shards}"
  )"
done

generation_gate="$(
  submit_job slurm/validate_generation_run.sbatch \
    --dependency="afterok:${generation_job}" \
    --export="ALL,GENERATION_NAME=teacher,GENERATION_ID=${generation_id},GENERATION_CONFIG=${generation_config},INPUT_NAME=${input_name},EXPECTED_PROMPTS=${train_size}"
)"

score_job=""
for ((pass = 1; pass <= score_passes; pass++)); do
  if [[ -z "${score_job}" ]]; then
    dependency="afterok:${generation_gate}"
  else
    dependency="afterany:${score_job}"
  fi
  score_job="$(
    submit_job slurm/score_teacher.sbatch \
      --array="0-$((num_gpu_shards - 1))%${num_gpu_shards}" \
      --time=12:00:00 \
      --dependency="${dependency}" \
      --export="ALL,GENERATION_JOB_ID=${generation_id},GENERATION_NAME=teacher,GENERATION_CONFIG=${generation_config},PROMPTS_NAME=${input_name},SCORE_CONCURRENCY=8,SCORE_NUM_SHARDS=${num_gpu_shards}"
  )"
done

score_gate="$(
  submit_job slurm/validate_score_run.sbatch \
    --dependency="afterok:${score_job}" \
    --export="ALL,GENERATION_NAME=teacher,GENERATION_ID=${generation_id},GENERATION_CONFIG=${generation_config},EXPECTED_PROMPTS=${train_size}"
)"
cluster_job="$(
  submit_job slurm/cluster_teacher.sbatch \
    --array="0-$((num_gpu_shards - 1))%${num_gpu_shards}" \
    --time=12:00:00 \
    --mem=96G \
    --dependency="afterok:${score_gate}" \
    --export="ALL,GENERATION_JOB_ID=${generation_id},TARGET_NAME=${target_name},CLUSTER_NUM_SHARDS=${num_gpu_shards}"
)"
merge_job="$(
  submit_job slurm/merge_teacher_clusters.sbatch \
    --dependency="afterok:${cluster_job}" \
    --export="ALL,GENERATION_ID=${generation_id},TARGET_NAME=${target_name},EXPECTED_PROMPTS=${train_size},CLUSTER_NUM_SHARDS=${num_gpu_shards}"
)"
target_gate="$(
  submit_job slurm/validate_teacher_targets.sbatch \
    --dependency="afterok:${merge_job}" \
    --time=02:00:00 \
    --mem=32G \
    --export="ALL,GENERATION_ID=${generation_id},TARGET_NAME=${target_name},PROMPTS_NAME=${input_name},EXPECTED_PROMPTS=${train_size}"
)"

printf '{"teacher_profile":"%s","train_size":%s,"bootstrap":"%s","generation_last":"%s","generation_gate":"%s","score_last":"%s","score_gate":"%s","cluster":"%s","merge":"%s","target_gate":"%s"}\n' \
  "${teacher_profile}" "${train_size}" "${bootstrap_job}" "${generation_job}" "${generation_gate}" "${score_job}" \
  "${score_gate}" "${cluster_job}" "${merge_job}" "${target_gate}"
