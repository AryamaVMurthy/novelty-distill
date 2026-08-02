#!/bin/bash
# Run the complete pinned official suite for one promoted checkpoint on four GPUs.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

eval_id="${EVAL_ID:?set EVAL_ID}"
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"
model_revision="${MODEL_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"
model_dtype="${MODEL_DTYPE:-auto}"
lora_name="${LORA_NAME:-}"
lora_path="${LORA_PATH:-}"
if [[ -n "${lora_path}" && -z "${lora_name}" ]]; then
  lora_name="novelty-model"
fi
dependency_job_id="${DEPENDENCY_JOB_ID:-}"
for value in "${eval_id}"; do
  if [[ -z "${value}" || "${value}" == */* || "${value}" == *..* || "${value}" == *,* ]]; then
    echo "EVAL_ID must be path-safe and comma-free" >&2
    exit 2
  fi
done
if [[ -n "${dependency_job_id}" && ! "${dependency_job_id}" =~ ^[0-9]+$ ]]; then
  echo "DEPENDENCY_JOB_ID must be numeric" >&2
  exit 2
fi
for value in "${model_path}" "${model_revision}"; do
  if [[ -z "${value}" || "${value}" == *,* ]]; then
    echo "MODEL_PATH and MODEL_REVISION must be non-empty and comma-free" >&2
    exit 2
  fi
done
if [[ -n "${lora_name}" || -n "${lora_path}" ]]; then
  if [[ -z "${lora_name}" || -z "${lora_path}" || "${lora_name}" == *,* || "${lora_path}" == *,* ]]; then
    echo "LORA_NAME and LORA_PATH must be set together and comma-free" >&2
    exit 2
  fi
  if [[ "${lora_name}" != novelty-model ]]; then
    echo "LORA_NAME must be novelty-model for the pinned official clients" >&2
    exit 2
  fi
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

dependency=()
if [[ -n "${dependency_job_id}" ]]; then
  dependency=(--dependency="afterok:${dependency_job_id}")
fi
common_export="ALL,EVAL_ID=${eval_id},MODEL_PATH=${model_path},MODEL_REVISION=${model_revision},MODEL_DTYPE=${model_dtype},NUM_GENERATIONS=10"
if [[ -n "${lora_name}" ]]; then
  common_export+=",LORA_NAME=${lora_name},LORA_PATH=${lora_path}"
fi
novelty_job="$(
  submit_job slurm/evaluate_official.sbatch \
    --time=12:00:00 --mem=96G \
    "${dependency[@]}" \
    --export="${common_export},EVAL_SUITE=noveltybench,EVAL_LIMIT=100"
)"
domains=(causal 3d boolean)
limits=(61 9 35)
hypospace_jobs=()
for index in "${!domains[@]}"; do
  hypospace_jobs+=("$(
    submit_job slurm/evaluate_official.sbatch \
      --time=12:00:00 --mem=64G \
      "${dependency[@]}" \
      --export="${common_export},EVAL_SUITE=hypospace,HYPOSPACE_DOMAIN=${domains[$index]},EVAL_LIMIT=${limits[$index]}"
  )")
done
suite_dependency="${novelty_job}:$(IFS=:; printf '%s' "${hypospace_jobs[*]}")"
combine_job="$(
  submit_job slurm/combine_official_results.sbatch \
    --dependency="afterok:${suite_dependency}" \
    --export="ALL,EVAL_ID=${eval_id}"
)"
printf '{"eval_id":"%s","noveltybench":"%s","hypospace":"%s","combined":"%s"}\n' \
  "${eval_id}" "${novelty_job}" "$(IFS=','; printf '%s' "${hypospace_jobs[*]}")" \
  "${combine_job}"
