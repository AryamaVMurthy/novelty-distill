#!/bin/bash
# Reuse a completed teacher subset before extending it to a larger frozen bank.

#SBATCH --job-name=nd-bootstrap-generation
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
source_generation_id="${SOURCE_GENERATION_ID:?set SOURCE_GENERATION_ID}"
target_generation_id="${TARGET_GENERATION_ID:?set TARGET_GENERATION_ID}"
input_name="${INPUT_NAME:?set INPUT_NAME}"
generation_config="${GENERATION_CONFIG:-configs/generation/teacher.yaml}"
for value in "${source_generation_id}" "${target_generation_id}" "${input_name}"; do
  if [[ -z "${value}" || "${value}" == */* || "${value}" == *..* ]]; then
    echo "generation IDs and input name must be path-safe" >&2
    exit 2
  fi
done

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
if [[ "${generation_config}" != /* ]]; then
  generation_config="${repo_dir}/${generation_config}"
fi
export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy

mkdir -p "${scratch_root}/locks"
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
flock -u 9
exec 8>"${scratch_root}/locks/venv-inference.lock"
flock -x 8
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

"${venv_dir}/bin/python" "${repo_dir}/scripts/bootstrap_generation_run.py" \
  --source-dir "${scratch_root}/generations/teacher/${source_generation_id}" \
  --target-dir "${scratch_root}/generations/teacher/${target_generation_id}" \
  --target-input "${scratch_root}/data/${input_name}" \
  --config "${generation_config}"
