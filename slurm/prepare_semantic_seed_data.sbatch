#!/bin/bash
#SBATCH --job-name=nd-seed-data
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/semantic-seed-runs}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/semantic-seed-data"

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
export HF_HOME="${scratch_root}/cache/huggingface"

mkdir -p "${scratch_root}/data" "${scratch_root}/cache" "${scratch_root}/venvs"
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  uv venv --python 3.12 "${venv_dir}"
fi
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/data.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

declare -a requests=(
  "train open 128 semantic-seed-train-128.jsonl"
  "test open 32 semantic-seed-eval-32.jsonl"
)
for request in "${requests[@]}"; do
  read -r split task size filename <<<"${request}"
  output="${scratch_root}/data/${filename}"
  if [[ ! -f "${output}" ]]; then
    "${venv_dir}/bin/python" "${repo_dir}/scripts/prepare_tomato.py" \
      --split "${split}" \
      --task "${task}" \
      --size "${size}" \
      --seed 17 \
      --output "${output}"
  fi
  test "$(wc -l <"${output}")" -eq "${size}"
  sha256sum "${output}"
done
