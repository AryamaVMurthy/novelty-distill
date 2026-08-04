#!/bin/bash
# Freeze a compact audit artifact from the eight corrected K=4 reruns.

#SBATCH --job-name=nd-summary-corrected
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
venv_dir="${scratch_root}/venvs/data"
evaluation_root="${scratch_root}/evaluations/tomato"
output="${evaluation_root}/corrected-k4-seed17-v2-summary.json"
d2_input="${D2_INPUT:-/scratch/node05/${USER}/novelty-distill/evaluations/tomato/D2-tomato1k-seed17-temporal-k4-corrected-v2.json}"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9

export UV_CACHE_DIR="${scratch_root}/cache/uv-analysis"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
mkdir -p "${scratch_root}/locks"
exec 8>"${scratch_root}/locks/venv-data.lock"
flock -x 8
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/data.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

"${venv_dir}/bin/python" "${repo_dir}/scripts/summarize_corrected_evaluations.py" \
  --input "A0=${evaluation_root}/A0-temporal-k4-from-k16-corrected-v2.json" \
  --input "A1=${evaluation_root}/A1-temporal-k4-from-k16-corrected-v2.json" \
  --input "B1=${evaluation_root}/B1-tomato1k-seed17-temporal-k4-corrected-v2.json" \
  --input "B2b=${evaluation_root}/B2b-tomato1k-seed17-temporal-k4-corrected-v2.json" \
  --input "C1-best1=${evaluation_root}/C1-best1-tomato1k-seed17-temporal-k4-corrected-v2.json" \
  --input "C2-best1=${evaluation_root}/C2-best1-tomato1k-seed17-temporal-k4-corrected-v2.json" \
  --input "D1=${evaluation_root}/D1-tomato1k-seed17-temporal-k4-corrected-v2.json" \
  --input "D2=${d2_input}" \
  --job A0=19055 --job A1=19061 --job B1=19056 --job B2b=19057 \
  --job C1-best1=19058 --job C2-best1=19059 --job D1=19060 --job D2=19062 \
  --output "${output}"

sha256sum "${output}"
