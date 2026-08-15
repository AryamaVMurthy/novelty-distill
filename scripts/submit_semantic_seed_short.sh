#!/bin/bash
# Submit training, generation, judging, clustering, and validity-first selection.

set -euo pipefail

target_job_id="${TARGET_JOB_ID:?set TARGET_JOB_ID to the completed target-preparation job}"
if ! [[ "${target_job_id}" =~ ^[0-9]+$ ]]; then
  echo "TARGET_JOB_ID must be numeric" >&2
  exit 2
fi

submit_job() {
  local output job_id
  output="$(REPO_REF=codex/semantic-seed-runs scripts/turing_submit.sh "$@")"
  printf '%s\n' "${output}" >&2
  job_id="$(awk '/Submitted batch job/{print $4}' <<<"${output}" | tail -1)"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || exit 1
  printf '%s\n' "${job_id}"
}

training_job="$(
  submit_job slurm/train_smoke.sbatch \
    --time=04:00:00 \
    --array=0-5%1 \
    --dependency="afterok:${target_job_id}" \
    --export=ALL,BASELINE_MATRIX=semantic_seed_short
)"
generation_job="$(
  submit_job slurm/generate_semantic_seed_short.sbatch \
    --array=0-5%1 \
    --dependency="afterok:${training_job}"
)"
score_job="$(
  submit_job slurm/score_semantic_seed_short.sbatch \
    --array=0-5%1 \
    --dependency="afterok:${generation_job}"
)"
cluster_job="$(
  submit_job slurm/cluster_semantic_seed_short.sbatch \
    --array=0-5%1 \
    --dependency="afterok:${score_job}"
)"
analysis_job="$(
  submit_job slurm/analyze_semantic_seed_short.sbatch \
    --dependency="afterok:${cluster_job}"
)"

printf '{"training":"%s","generation":"%s","score":"%s","cluster":"%s","analysis":"%s"}\n' \
  "${training_job}" "${generation_job}" "${score_job}" "${cluster_job}" "${analysis_job}"
