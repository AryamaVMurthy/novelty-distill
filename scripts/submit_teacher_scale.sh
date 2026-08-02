#!/bin/bash
# Submit a fail-closed four-GPU teacher bank for TOMATO 5k or 20k.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

train_size="${TRAIN_SIZE:-5000}"
generation_passes="${GENERATION_PASSES:-2}"
score_passes="${SCORE_PASSES:-2}"
num_gpu_shards="${NUM_GPU_SHARDS:-4}"
case "${train_size}" in
  5000|20000) ;;
  *) echo "TRAIN_SIZE must be 5000 or 20000" >&2; exit 2 ;;
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

generation_id="teacher-${train_size}-v1"
input_name="tomato-open-train-${train_size}.jsonl"
target_name="teacher-targets-tomato${train_size}-v1.json"

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

generation_job=""
for ((pass = 1; pass <= generation_passes; pass++)); do
  dependency=()
  if [[ -n "${generation_job}" ]]; then
    dependency=(--dependency="afterany:${generation_job}")
  fi
  generation_job="$(
    submit_job slurm/sglang_smoke.sbatch \
      --array="0-$((num_gpu_shards - 1))%${num_gpu_shards}" \
      --time=12:00:00 \
      "${dependency[@]}" \
      --export="ALL,GENERATION_CONFIG=configs/generation/teacher.yaml,OUTPUT_NAME=teacher,OUTPUT_RUN_ID=${generation_id},INPUT_NAME=${input_name},INPUT_LIMIT=${train_size},GENERATION_CONCURRENCY=8,GENERATION_NUM_SHARDS=${num_gpu_shards}"
  )"
done

generation_gate="$(
  submit_job slurm/validate_generation_run.sbatch \
    --dependency="afterok:${generation_job}" \
    --export="ALL,GENERATION_NAME=teacher,GENERATION_ID=${generation_id},GENERATION_CONFIG=configs/generation/teacher.yaml,INPUT_NAME=${input_name},EXPECTED_PROMPTS=${train_size}"
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
      --export="ALL,GENERATION_JOB_ID=${generation_id},GENERATION_NAME=teacher,GENERATION_CONFIG=configs/generation/teacher.yaml,PROMPTS_NAME=${input_name},SCORE_CONCURRENCY=8,SCORE_NUM_SHARDS=${num_gpu_shards}"
  )"
done

score_gate="$(
  submit_job slurm/validate_score_run.sbatch \
    --dependency="afterok:${score_job}" \
    --export="ALL,GENERATION_NAME=teacher,GENERATION_ID=${generation_id},GENERATION_CONFIG=configs/generation/teacher.yaml,EXPECTED_PROMPTS=${train_size}"
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

printf '{"train_size":%s,"generation_last":"%s","generation_gate":"%s","score_last":"%s","score_gate":"%s","cluster":"%s","merge":"%s","target_gate":"%s"}\n' \
  "${train_size}" "${generation_job}" "${generation_gate}" "${score_job}" \
  "${score_gate}" "${cluster_job}" "${merge_job}" "${target_gate}"
