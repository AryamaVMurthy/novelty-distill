#!/bin/bash
# Copy one completed final adapter between node-local scratch roots, verify its
# byte identity, and release a held evaluation root only after the gate passes.

#SBATCH --job-name=nd-stage-final-adapter
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

source_node="${SOURCE_NODE:?SOURCE_NODE is required}"
baseline_id="${BASELINE_ID:?BASELINE_ID is required}"
train_seed="${TRAIN_SEED:?TRAIN_SEED is required}"
release_job_id="${RELEASE_JOB_ID:?RELEASE_JOB_ID is required}"

if [[ ! "${source_node}" =~ ^node[0-9]{2}$ ]]; then
  echo "SOURCE_NODE must have the form nodeNN" >&2
  exit 2
fi
if [[ ! "${baseline_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "BASELINE_ID is not path-safe" >&2
  exit 2
fi
if [[ ! "${train_seed}" =~ ^[0-9]+$ ]]; then
  echo "TRAIN_SEED must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${release_job_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RELEASE_JOB_ID must be a Slurm job ID" >&2
  exit 2
fi
if [[ "$(hostname)" == "${source_node}" ]]; then
  echo "staging destination must differ from SOURCE_NODE" >&2
  exit 2
fi

relative="checkpoints/${baseline_id}-tomato1k-seed${train_seed}/final"
source_root="/scratch/${source_node}/${USER}/novelty-distill"
destination_root="/scratch/${USER}/novelty-distill"
destination="${destination_root}/${relative}"
staging="${destination}.staging-${SLURM_JOB_ID}"

artifact_identity() {
  PYTHONPATH="${destination_root}/repo/src" \
    "${destination_root}/venvs/inference/bin/python" \
    "${destination_root}/repo/scripts/hash_model_artifact.py" \
    --path "$1"
}

source_identity="$(
  ssh -o BatchMode=yes turing \
    "PYTHONPATH='${source_root}/repo/src' '${source_root}/venvs/inference/bin/python' '${source_root}/repo/scripts/hash_model_artifact.py' --path '${source_root}/${relative}'"
)"

mkdir -p "$(dirname "${destination}")"
if [[ -e "${destination}" ]]; then
  destination_identity="$(artifact_identity "${destination}")"
  if [[ "${destination_identity}" != "${source_identity}" ]]; then
    echo "existing destination adapter does not match the source" >&2
    exit 1
  fi
  copy_status="reused"
else
  if [[ -e "${staging}" ]]; then
    echo "unique staging path already exists: ${staging}" >&2
    exit 1
  fi
  mkdir "${staging}"
  rsync -a --partial --info=stats1 \
    "turing:${source_root}/${relative}/" \
    "${staging}/"
  test -s "${staging}/adapter_config.json"
  test -s "${staging}/adapter_model.safetensors"
  destination_identity="$(artifact_identity "${staging}")"
  if [[ "${destination_identity}" != "${source_identity}" ]]; then
    echo "staged adapter does not match the source" >&2
    exit 1
  fi
  mv "${staging}" "${destination}"
  copy_status="created"
fi

scontrol release "${release_job_id}"
state="$(squeue -h -j "${release_job_id}" -o '%T|%R' || true)"
printf \
  'adapter_stage=complete status=%s baseline=%s seed=%s source=%s target=%s artifact_identity=%s released_job=%s state=%s\n' \
  "${copy_status}" "${baseline_id}" "${train_seed}" "${source_node}" "$(hostname)" \
  "${source_identity}" "${release_job_id}" "${state:-not_in_queue}"
