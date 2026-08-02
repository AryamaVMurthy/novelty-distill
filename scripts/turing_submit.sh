#!/bin/bash
# Submit one repository Slurm script on Turing's node-local scratch.

set -euo pipefail

if (($# < 1)); then
  echo "usage: $0 slurm/JOB.sbatch [sbatch options]" >&2
  exit 2
fi
job_script="$1"
shift
if [[ "${job_script}" != slurm/*.sbatch || "${job_script}" == *..* ]]; then
  echo "job must be a slurm/*.sbatch path without '..'" >&2
  exit 2
fi

repo_ref="${REPO_REF:-codex/implementation}"
if [[ ! -f "${job_script}" ]]; then
  echo "missing local batch script: ${job_script}" >&2
  exit 1
fi

script_name="$(basename "${job_script}")"
remote_stage=".cache/novelty-distill-submit/${script_name}"
ssh turing 'mkdir -p "$HOME/.cache/novelty-distill-submit"'
scp -q "${job_script}" "turing:${remote_stage}"

printf -v quoted_ref '%q' "${repo_ref}"
printf -v quoted_stage '%q' "${remote_stage}"
remote_command="set -euo pipefail; export REPO_REF=${quoted_ref}"
remote_command+='; scratch_root=/scratch/$USER/novelty-distill; sbatch --output="$scratch_root/logs/slurm-%x-%j.out" --error="$scratch_root/logs/slurm-%x-%j.err"'
for argument in "$@"; do
  printf -v quoted_argument '%q' "${argument}"
  remote_command+=" ${quoted_argument}"
done
remote_command+=" ${quoted_stage}"
ssh turing "${remote_command}"
