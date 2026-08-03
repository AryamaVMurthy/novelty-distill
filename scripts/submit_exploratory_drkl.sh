#!/bin/bash
# Submit the two post-freeze DRKL treatments without changing the primary matrix.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

start_after_job_id="${START_AFTER_JOB_ID:-}"
if [[ -n "${start_after_job_id}" && ! "${start_after_job_id}" =~ ^[0-9]+$ ]]; then
  echo "START_AFTER_JOB_ID must be a numeric Slurm job ID" >&2
  exit 2
fi

dependency=()
if [[ -n "${start_after_job_id}" ]]; then
  dependency=(--dependency="afterok:${start_after_job_id}")
fi

jobs=()
for baseline_id in F1-best1 F1-diverse4; do
  output="$(
    scripts/turing_submit.sh slurm/train_smoke.sbatch \
      --time=06:00:00 \
      --mem=116G \
      --nice=10000 \
      "${dependency[@]}" \
      --export="ALL,TRAINING_BACKEND=trl,RUN_CONFIG=configs/training/gkd_tomato1k.yaml,BASELINE_ID=${baseline_id},BASELINE_REGISTRY=configs/exploratory_baselines.yaml,RUN_SUFFIX=exploratory-drkl-tomato1k-seed17"
  )"
  printf '%s\n' "${output}" >&2
  job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job/{print $4}' | tail -1)"
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "could not parse submitted job ID for ${baseline_id}" >&2
    exit 1
  fi
  jobs+=("${baseline_id}:${job_id}")
done

printf '{"treatment":"DRKL-gamma0.5","status":"secondary_exploratory","jobs":"%s"}\n' \
  "$(IFS=','; printf '%s' "${jobs[*]}")"
