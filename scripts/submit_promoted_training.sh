#!/bin/bash
# Submit only the baseline indexes promoted by the completed 1k/5k analysis.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

train_size="${TRAIN_SIZE:?set TRAIN_SIZE to 5000 or 20000}"
model_profile="${MODEL_PROFILE:-main}"
promoted_indices="${PROMOTED_INDICES:?set comma-separated baseline matrix indexes}"
target_gate_job_id="${TARGET_GATE_JOB_ID:?set the validated teacher-target job ID}"
submission_manifest="${SUBMISSION_MANIFEST:-}"
case "${model_profile}:${train_size}" in
  replication:1000)
    seeds="${TRAIN_SEEDS:-17}"
    training_passes="${TRAINING_PASSES:-2}"
    ;;
  main:5000|replication:5000)
    seeds="${TRAIN_SEEDS:-17}"
    training_passes="${TRAINING_PASSES:-5}"
    ;;
  main:20000|replication:20000)
    seeds="${TRAIN_SEEDS:-17,29,43}"
    training_passes="${TRAINING_PASSES:-17}"
    ;;
  *) echo "MODEL_PROFILE/TRAIN_SIZE combination is unsupported" >&2; exit 2 ;;
esac
if ! [[ "${target_gate_job_id}" =~ ^[0-9]+$ ]]; then
  echo "TARGET_GATE_JOB_ID must be numeric" >&2
  exit 2
fi
if ! [[ "${training_passes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAINING_PASSES must be positive" >&2
  exit 2
fi

declare -A seen=()
adapter_indices=()
run_gem=0
run_distillm=0
IFS=',' read -r -a requested <<<"${promoted_indices}"
for index in "${requested[@]}"; do
  if ! [[ "${index}" =~ ^([0-9]|1[0-8])$ ]] || [[ -n "${seen[$index]:-}" ]]; then
    echo "PROMOTED_INDICES must contain unique indexes from 0 through 18" >&2
    exit 2
  fi
  seen[$index]=1
  if [[ "${index}" == 5 ]]; then
    run_gem=1
  elif [[ "${index}" == 12 ]]; then
    run_distillm=1
  else
    adapter_indices+=("${index}")
  fi
done
if ((${#seen[@]} == 0)); then
  echo "PROMOTED_INDICES cannot be empty" >&2
  exit 2
fi
IFS=',' read -r -a seed_values <<<"${seeds}"
declare -A seen_seed=()
for seed in "${seed_values[@]}"; do
  if [[ "${seed}" != 17 && "${seed}" != 29 && "${seed}" != 43 ]] || \
    [[ -n "${seen_seed[$seed]:-}" ]]
  then
    echo "TRAIN_SEEDS must contain unique values from 17,29,43" >&2
    exit 2
  fi
  seen_seed[$seed]=1
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

adapter_spec=""
if ((${#adapter_indices[@]} > 0)); then
  adapter_spec="$(IFS=','; printf '%s' "${adapter_indices[*]}")"
fi
manifest_dependencies=()
for seed in "${seed_values[@]}"; do
  adapter_job=""
  if [[ -n "${adapter_spec}" ]]; then
    for ((pass = 1; pass <= training_passes; pass++)); do
      if [[ -z "${adapter_job}" ]]; then
        dependency="afterok:${target_gate_job_id}"
      else
        dependency="afterany:${adapter_job}"
      fi
      adapter_job="$(
        submit_job slurm/train_smoke.sbatch \
          --array="${adapter_spec}%4" \
          --time=12:00:00 \
          --mem=128G \
          --dependency="${dependency}" \
          --export="ALL,BASELINE_MATRIX=tomato_scale,TRAIN_SIZE=${train_size},TRAIN_SEED=${seed},MODEL_PROFILE=${model_profile}"
      )"
    done
    for index in "${adapter_indices[@]}"; do
      manifest_dependencies+=(
        --dependency "${index}:${seed}=${adapter_job}_${index}"
      )
    done
  fi
  if [[ "${run_gem}" == 1 ]]; then
    gem_job="$(
      submit_job slurm/train_smoke.sbatch \
        --array=5 \
        --time=12:00:00 \
        --mem=128G \
        --dependency="afterok:${target_gate_job_id}" \
        --export="ALL,BASELINE_MATRIX=tomato_scale,TRAIN_SIZE=${train_size},TRAIN_SEED=${seed},MODEL_PROFILE=${model_profile}"
    )"
    manifest_dependencies+=(--dependency "5:${seed}=${gem_job}")
  fi
  if [[ "${run_distillm}" == 1 ]]; then
    distillm_job="$(
      submit_job slurm/train_smoke.sbatch \
        --array=12 \
        --time=12:00:00 \
        --mem=192G \
        --gres=gpu:4 \
        --dependency="afterok:${target_gate_job_id}" \
        --export="ALL,BASELINE_MATRIX=tomato_scale,TRAIN_SIZE=${train_size},TRAIN_SEED=${seed},MODEL_PROFILE=${model_profile}"
    )"
    manifest_dependencies+=(--dependency "12:${seed}=${distillm_job}")
  fi
done
manifest_command=(
  uv run python scripts/render_promoted_training_manifest.py
  --model-profile "${model_profile}"
  --train-size "${train_size}"
  --indices "${promoted_indices}"
  --seeds "${seeds}"
  --target-gate-job-id "${target_gate_job_id}"
  "${manifest_dependencies[@]}"
)
if [[ -n "${submission_manifest}" ]]; then
  manifest_command+=(--output "${submission_manifest}")
fi
"${manifest_command[@]}"
