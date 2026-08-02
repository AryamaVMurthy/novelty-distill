#!/bin/bash
# Submit the resumable teacher-target and complete TOMATO-1k training chain on Turing.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

generation_run_id="${GENERATION_RUN_ID:-teacher-1k-v1}"
generation_passes="${GENERATION_PASSES:-2}"
score_passes="${SCORE_PASSES:-2}"
submit_training="${SUBMIT_TRAINING:-1}"
training_passes="${TRAINING_PASSES:-2}"
for name in "${generation_run_id}"; do
  if [[ -z "${name}" || "${name}" == */* || "${name}" == *..* ]]; then
    echo "generation run ID must be a non-empty path-safe name" >&2
    exit 2
  fi
done
for value in "${generation_passes}" "${score_passes}" "${training_passes}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "generation, score, and training passes must be positive integers" >&2
    exit 2
  fi
done
if [[ "${submit_training}" != 0 && "${submit_training}" != 1 ]]; then
  echo "SUBMIT_TRAINING must be 0 or 1" >&2
  exit 2
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

generation_job=""
for ((pass = 1; pass <= generation_passes; pass++)); do
  dependency=()
  if [[ -n "${generation_job}" ]]; then
    dependency=(--dependency="afterany:${generation_job}")
  fi
  generation_job="$(
    submit_job slurm/sglang_smoke.sbatch \
      --time=06:00:00 \
      "${dependency[@]}" \
      --export="ALL,GENERATION_CONFIG=configs/generation/teacher.yaml,OUTPUT_NAME=teacher,OUTPUT_RUN_ID=${generation_run_id},INPUT_LIMIT=1000,GENERATION_CONCURRENCY=8"
  )"
done

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
      --export="ALL,GENERATION_JOB_ID=${generation_run_id},GENERATION_NAME=teacher,GENERATION_CONFIG=configs/generation/teacher.yaml,SCORE_CONCURRENCY=8"
  )"
done

cluster_job="$(
  submit_job slurm/cluster_teacher.sbatch \
    --dependency="afterok:${score_job}" \
    --export="ALL,GENERATION_JOB_ID=${generation_run_id},TARGET_NAME=teacher-targets-tomato1k-v1.json"
)"

training_job=""
gem_training_job=""
distillm_training_job=""
if [[ "${submit_training}" == 1 ]]; then
  for ((pass = 1; pass <= training_passes; pass++)); do
    if [[ -z "${training_job}" ]]; then
      training_dependency="afterok:${cluster_job}"
    else
      training_dependency="afterany:${training_job}"
    fi
    training_job="$(
      submit_job slurm/train_smoke.sbatch \
        --time=06:00:00 \
        --mem=128G \
        --array=0-4,6-11,13-18%2 \
        --dependency="${training_dependency}" \
        --export=ALL,BASELINE_MATRIX=tomato1k
    )"
  done
  gem_training_job="$(
    submit_job slurm/train_smoke.sbatch \
      --time=06:00:00 \
      --mem=128G \
      --array=5 \
      --dependency="afterok:${cluster_job}" \
      --export=ALL,BASELINE_MATRIX=tomato1k
  )"
  distillm_training_job="$(
    submit_job slurm/train_smoke.sbatch \
      --time=06:00:00 \
      --mem=192G \
      --gres=gpu:4 \
      --array=12 \
      --dependency="afterok:${cluster_job}" \
      --export=ALL,BASELINE_MATRIX=tomato1k
  )"
fi

printf '{"generation_last":"%s","score_last":"%s","cluster":"%s","adapter_training_final":"%s","adapter_training_passes":%s,"gem_training":"%s","distillm_training":"%s"}\n' \
  "${generation_job}" "${score_job}" "${cluster_job}" "${training_job}" \
  "${training_passes}" "${gem_training_job}" "${distillm_training_job}"
