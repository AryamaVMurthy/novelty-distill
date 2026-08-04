#!/bin/bash
# Preserve a stale staged checkout and replace it with a clean branch clone.

#SBATCH --job-name=nd-repair-repo
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
preserved_dir="${scratch_root}/repo-stage-stale-18775"
repo_ref="${REPO_REF:-codex/implementation}"
expected_commit="${EXPECTED_COMMIT:?set EXPECTED_COMMIT}"

test -d "${repo_dir}/.git"
test ! -e "${preserved_dir}"
mv "${repo_dir}" "${preserved_dir}"
git clone --branch "${repo_ref}" --single-branch \
  https://github.com/AryamaVMurthy/novelty-distill.git "${repo_dir}"
test "$(git -C "${repo_dir}" rev-parse HEAD)" = "${expected_commit}"
test -z "$(git -C "${repo_dir}" status --porcelain)"
echo "repository_repair=complete commit=${expected_commit} preserved=${preserved_dir}"
