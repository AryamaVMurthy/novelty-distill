#!/bin/bash
#SBATCH --job-name=nd-cluster-teacher
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

module load u22/cuda/12.4

generation_job_id="${GENERATION_JOB_ID:?set GENERATION_JOB_ID to the teacher SGLang job}"
target_name="${TARGET_NAME:-teacher-targets.json}"
if [[ "${target_name}" == */* || "${target_name}" == *..* ]]; then
  echo "TARGET_NAME must be a filename without '..'" >&2
  exit 2
fi
repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_LINK_MODE=copy
export HF_HOME="${scratch_root}/cache/huggingface"
export XDG_CACHE_HOME="${scratch_root}/cache/xdg"
export TORCH_EXTENSIONS_DIR="${scratch_root}/cache/torch-extensions"
export TRITON_CACHE_DIR="${scratch_root}/cache/triton"
export CUDA_CACHE_PATH="${scratch_root}/cache/cuda"
export TMPDIR="${scratch_root}/tmp"

mkdir -p "${scratch_root}/tmp"
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/inference.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

"${venv_dir}/bin/python" "${repo_dir}/scripts/cluster_teacher_generations.py" \
  --score-dir "${scratch_root}/teacher-scores/${generation_job_id}" \
  --output "${scratch_root}/data/teacher-generations-${generation_job_id}.jsonl" \
  --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml"

"${venv_dir}/bin/python" "${repo_dir}/scripts/derive_teacher_targets.py" \
  --input "${scratch_root}/data/teacher-generations-${generation_job_id}.jsonl" \
  --output "${scratch_root}/data/${target_name}" \
  --seed 17
