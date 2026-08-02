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
printf -v quoted_ref '%q' "${repo_ref}"
remote_command="set -euo pipefail; repo_ref=${quoted_ref}"
remote_command+='; scratch_root=/scratch/$USER/novelty-distill; repo_dir=$scratch_root/repo; mkdir -p "$scratch_root/logs"; exec 9>"$repo_dir/.git/novelty-distill-sync.lock"; flock -x 9; git -C "$repo_dir" diff --quiet; git -C "$repo_dir" fetch origin "$repo_ref"; git -C "$repo_dir" switch "$repo_ref"; git -C "$repo_dir" merge --ff-only "origin/$repo_ref"; flock -u 9; cd "$repo_dir"; sbatch --output="$scratch_root/logs/slurm-%x-%j.out" --error="$scratch_root/logs/slurm-%x-%j.err"'
for argument in "$@"; do
  printf -v quoted_argument '%q' "${argument}"
  remote_command+=" ${quoted_argument}"
done
printf -v quoted_job '%q' "${job_script}"
remote_command+=" ${quoted_job}"
printf -v quoted_remote '%q' "${remote_command}"

ssh turing "srun --partition=u22 --qos=high --account=priyesh.shukla --nodelist=node01 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:03:00 bash -lc ${quoted_remote}"
