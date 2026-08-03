#!/bin/bash
#SBATCH --job-name=nd-hypo-data
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
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
official_dir="${scratch_root}/official/hypospace"
venv_dir="${scratch_root}/venvs/hypospace"

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
export TMPDIR="${scratch_root}/tmp"

mkdir -p "${scratch_root}/tmp"
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
test "$(git -C "${official_dir}" rev-parse HEAD)" = \
  "c69e9318577b34b5b896996571aefd4ba6053f58"

uv venv --python 3.12 --allow-existing "${venv_dir}"
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/hypospace.lock"

mkdir -p \
  "${official_dir}/causal/datasets/node03" \
  "${official_dir}/3d/datasets" \
  "${official_dir}/boolean/datasets"

if [[ ! -s "${official_dir}/causal/datasets/node03/n3_all_observations.json" ]]; then
  (
    cd "${official_dir}/causal"
    "${venv_dir}/bin/python" generate_causal_dataset.py \
      --nodes 3 \
      --seed 33550336 \
      --output datasets/node03/n3_all_observations.json
  )
fi

if [[ ! -s "${official_dir}/3d/datasets/3d_grid3_h3.json" ]]; then
  (
    cd "${official_dir}/3d"
    "${venv_dir}/bin/python" generate_3d_dataset_complete.py \
      --grid-size 3 \
      --max-height 3 \
      --max-blocks 1 \
      --fixed \
      --seed 33550336 \
      --output datasets/3d_grid3_h3.json
  )
fi

if [[ ! -s "${official_dir}/boolean/datasets/boolean_2var.json" ]]; then
  (
    cd "${official_dir}/boolean"
    "${venv_dir}/bin/python" boolean_dataset.py \
      --operators basic \
      --max-depth 2 \
      --output datasets/boolean_2var.json \
      --seed 33550336
  )
fi

sha256sum \
  "${official_dir}/causal/datasets/node03/n3_all_observations.json" \
  "${official_dir}/3d/datasets/3d_grid3_h3.json" \
  "${official_dir}/boolean/datasets/boolean_2var.json"
