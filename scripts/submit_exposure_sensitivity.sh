#!/bin/bash
# Submit the post-primary 4x-exposure controls required for diverse-4 interpretation.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

start_after_job_id="${START_AFTER_JOB_ID:?set the frozen primary controller job ID}"
training_passes="${TRAINING_PASSES:-2}"
if ! [[ "${start_after_job_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "START_AFTER_JOB_ID must be a positive numeric Slurm job ID" >&2
  exit 2
fi
if ! [[ "${training_passes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAINING_PASSES must be positive" >&2
  exit 2
fi

submit_job() {
  local output job_id
  output="$(scripts/turing_submit.sh "$@")"
  printf '%s\n' "${output}" >&2
  job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job/{print $4}' | tail -1)"
  if ! [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "could not parse submitted exposure-sensitivity job ID" >&2
    exit 1
  fi
  printf '%s\n' "${job_id}"
}

training_job=""
for ((pass = 1; pass <= training_passes; pass++)); do
  if [[ -z "${training_job}" ]]; then
    dependency="afterok:${start_after_job_id}"
  else
    dependency="afterany:${training_job}"
  fi
  training_job="$(
    submit_job slurm/train_smoke.sbatch \
      --array=0-4%2 \
      --time=06:00:00 \
      --mem=116G \
      --nice=10000 \
      --dependency="${dependency}" \
      --export="ALL,BASELINE_MATRIX=exposure1k"
  )"
done

printf '{"study":"tomato1k-exposure4x","status":"secondary_sensitivity","training_final":"%s","passes":%s}\n' \
  "${training_job}" "${training_passes}"
