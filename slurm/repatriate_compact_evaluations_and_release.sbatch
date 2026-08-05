#!/bin/bash
# Repatriate the declared compact-evaluation JSON files to node01-local scratch,
# byte-verify every copy, and release aggregate analysis only after all pass.

#SBATCH --job-name=nd-repatriate-evals
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

source_node="${SOURCE_NODE:?SOURCE_NODE is required}"
release_job_id="${RELEASE_JOB_ID:?RELEASE_JOB_ID is required}"
evaluation_specs="${EVALUATION_SPECS:?EVALUATION_SPECS is required}"

if [[ ! "${source_node}" =~ ^node[0-9]{2}$ ]]; then
  echo "SOURCE_NODE must have the form nodeNN" >&2
  exit 2
fi
if [[ ! "${release_job_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RELEASE_JOB_ID must be a Slurm job ID" >&2
  exit 2
fi
if [[ "$(hostname)" == "${source_node}" ]]; then
  echo "repatriation destination must differ from SOURCE_NODE" >&2
  exit 2
fi

source_root="/scratch/${source_node}/${USER}/novelty-distill"
destination_root="/scratch/${USER}/novelty-distill"
IFS=':' read -r -a specs <<<"${evaluation_specs}"
if ((${#specs[@]} == 0)); then
  echo "EVALUATION_SPECS must declare at least one method@seed pair" >&2
  exit 2
fi

copied=0
reused=0
declare -a verified=()
for spec in "${specs[@]}"; do
  if [[ ! "${spec}" =~ ^([A-Za-z0-9][A-Za-z0-9._-]*)@([0-9]+)$ ]]; then
    echo "invalid evaluation spec: ${spec}" >&2
    exit 2
  fi
  baseline_id="${BASH_REMATCH[1]}"
  train_seed="${BASH_REMATCH[2]}"
  relative="evaluations/tomato/${baseline_id}-tomato1k-seed${train_seed}-temporal-k4-corrected-v2.json"
  source="${source_root}/${relative}"
  destination="${destination_root}/${relative}"
  staging="${destination}.staging-${SLURM_JOB_ID}"
  source_hash="$(ssh -o BatchMode=yes turing "sha256sum '${source}'" | cut -d' ' -f1)"

  mkdir -p "$(dirname "${destination}")"
  if [[ -e "${destination}" ]]; then
    destination_hash="$(sha256sum "${destination}" | cut -d' ' -f1)"
    if [[ "${destination_hash}" != "${source_hash}" ]]; then
      echo "existing evaluation does not match source: ${relative}" >&2
      exit 1
    fi
    reused=$((reused + 1))
  else
    if [[ -e "${staging}" ]]; then
      echo "unique staging path already exists: ${staging}" >&2
      exit 1
    fi
    rsync -a --partial --info=stats1 "turing:${source}" "${staging}"
    test -s "${staging}"
    destination_hash="$(sha256sum "${staging}" | cut -d' ' -f1)"
    if [[ "${destination_hash}" != "${source_hash}" ]]; then
      echo "staged evaluation does not match source: ${relative}" >&2
      exit 1
    fi
    mv "${staging}" "${destination}"
    copied=$((copied + 1))
  fi
  verified+=("${spec}=${source_hash}")
done

scontrol release "${release_job_id}"
state="$(squeue -h -j "${release_job_id}" -o '%T|%R' || true)"
printf \
  'evaluation_repatriation=complete source=%s target=%s copied=%s reused=%s released_job=%s state=%s verified=%s\n' \
  "${source_node}" "$(hostname)" "${copied}" "${reused}" "${release_job_id}" \
  "${state:-not_in_queue}" "$(IFS=,; printf '%s' "${verified[*]}")"
