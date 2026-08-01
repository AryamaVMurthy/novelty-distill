#!/bin/bash
#SBATCH --job-name=nd-prepare
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_url="${REPO_URL:-https://github.com/AryamaVMurthy/novelty-distill.git}"
repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export HF_HOME="${scratch_root}/cache/huggingface"

mkdir -p "${scratch_root}" "${scratch_root}/data" "${scratch_root}/cache"
if [[ ! -d "${repo_dir}/.git" ]]; then
  git clone --branch "${repo_ref}" --single-branch "${repo_url}" "${repo_dir}"
else
  git -C "${repo_dir}" diff --quiet
  git -C "${repo_dir}" fetch origin "${repo_ref}"
  git -C "${repo_dir}" switch "${repo_ref}"
  git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  uv venv --python 3.12 "${venv_dir}"
fi
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/data.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

output="${scratch_root}/data/tomato-open-train-1000.jsonl"
if [[ ! -f "${output}" ]]; then
  "${venv_dir}/bin/python" "${repo_dir}/scripts/prepare_tomato.py" \
    --split train \
    --task open \
    --size 1000 \
    --seed 17 \
    --output "${output}"
fi

wc -l "${output}"
sha256sum "${output}"
