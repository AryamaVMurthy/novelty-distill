#!/bin/bash
# Validate and combine one model's NoveltyBench and three HypoSpace summaries.

#SBATCH --job-name=nd-combine-official
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

eval_id="${EVAL_ID:?set EVAL_ID}"
if [[ "${eval_id}" == */* || "${eval_id}" == *..* ]]; then
  echo "EVAL_ID must be path-safe" >&2
  exit 2
fi
repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
result_root="${scratch_root}/evaluations/official/${eval_id}"
export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
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

"${venv_dir}/bin/python" "${repo_dir}/scripts/combine_official_results.py" \
  --eval-id "${eval_id}" \
  --input "${result_root}/noveltybench/summary.json" \
  --input "${result_root}/hypospace/causal/summary.json" \
  --input "${result_root}/hypospace/3d/summary.json" \
  --input "${result_root}/hypospace/boolean/summary.json" \
  --output "${result_root}/combined.json"
test -s "${result_root}/combined.json"
