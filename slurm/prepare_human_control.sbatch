#!/bin/bash
# Prepare deterministic A3 historical-target generation shards on node-local scratch.

#SBATCH --job-name=nd-prepare-a3
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
input_path="${scratch_root}/data/tomato-open-test-1658.jsonl"
output_dir="${scratch_root}/generations/evaluation-student/A3-temporal-k1"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
flock -u 9

export UV_CACHE_DIR="${scratch_root}/cache/uv-a3"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
export HF_HOME="${scratch_root}/cache/huggingface"
mkdir -p "${scratch_root}/locks"
exec 8>"${scratch_root}/locks/venv-inference.lock"
flock -x 8
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

"${venv_dir}/bin/python" "${repo_dir}/scripts/create_human_control_generations.py" \
  --input "${input_path}" \
  --config "${repo_dir}/configs/generation/eval_historical.yaml" \
  --output-dir "${output_dir}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/create_human_control_generations.py" \
  --input "${input_path}" \
  --config "${repo_dir}/configs/generation/eval_historical.yaml" \
  --output-dir "${output_dir}"
test "$(find "${output_dir}" -maxdepth 1 -name '*.json' | wc -l)" -eq 1658
