#!/bin/bash
# Build the unique-prompt, eight-method, method-balanced-repeat judge packet.

#SBATCH --job-name=nd-prepare-deepinfra
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

repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"
score_root="${scratch_root}/evaluation-scores"
output_dir="${scratch_root}/evaluations/independent-judge"
packet_path="${output_dir}/deepinfra-calibration-packet-20260805-v3.json"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9

export UV_CACHE_DIR="${scratch_root}/cache/uv-analysis"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${scratch_root}/locks" "${output_dir}"
exec 8>"${scratch_root}/locks/venv-data.lock"
flock -x 8
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/data.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

test ! -e "${packet_path}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/prepare_independent_judge_calibration.py" \
  --prompts "${scratch_root}/data/tomato-composition-test-1658.jsonl" \
  --method "A0=${score_root}/A0-temporal-k16-seed17000" \
  --method "A1=${score_root}/A1-temporal-k16-seed17000" \
  --method "B2a=${score_root}/B2a-tomato1k-seed17-temporal-k4" \
  --method "B2b=${score_root}/B2b-tomato1k-seed17-temporal-k4" \
  --method "C1-best1=${score_root}/C1-best1-tomato1k-seed17-temporal-k4" \
  --method "C2-best1=${score_root}/C2-best1-tomato1k-seed17-temporal-k4" \
  --method "D1=${score_root}/D1-tomato1k-seed17-temporal-k4" \
  --method "D2=${score_root}/D2-tomato1k-seed17-temporal-k4" \
  --output "${packet_path}" \
  --samples-per-method 60 \
  --repeat-fraction 0.10 \
  --max-sample-index 4 \
  --seed 20260805

test -s "${packet_path}"
sha256sum "${packet_path}"
