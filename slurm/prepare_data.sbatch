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
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
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
"${venv_dir}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/data.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

for task in open composition; do
  for size in 1000 5000 20000; do
    output="${scratch_root}/data/tomato-${task}-train-${size}.jsonl"
    if [[ ! -f "${output}" ]]; then
      "${venv_dir}/bin/python" "${repo_dir}/scripts/prepare_tomato.py" \
        --split train \
        --task "${task}" \
        --size "${size}" \
        --seed 17 \
        --output "${output}"
    fi
    wc -l "${output}"
    sha256sum "${output}"
  done
done

for task in open composition; do
  output="${scratch_root}/data/tomato-${task}-test-1658.jsonl"
  if [[ ! -f "${output}" ]]; then
    "${venv_dir}/bin/python" "${repo_dir}/scripts/prepare_tomato.py" \
      --split test \
      --task "${task}" \
      --size 1658 \
      --seed 17 \
      --output "${output}"
  fi
  wc -l "${output}"
  sha256sum "${output}"
done

"${venv_dir}/bin/python" "${repo_dir}/scripts/validate_prepared_tomato.py" \
  --train-open "1000=${scratch_root}/data/tomato-open-train-1000.jsonl" \
  --train-open "5000=${scratch_root}/data/tomato-open-train-5000.jsonl" \
  --train-open "20000=${scratch_root}/data/tomato-open-train-20000.jsonl" \
  --train-composition "1000=${scratch_root}/data/tomato-composition-train-1000.jsonl" \
  --train-composition "5000=${scratch_root}/data/tomato-composition-train-5000.jsonl" \
  --train-composition "20000=${scratch_root}/data/tomato-composition-train-20000.jsonl" \
  --test-open "${scratch_root}/data/tomato-open-test-1658.jsonl" \
  --test-composition "${scratch_root}/data/tomato-composition-test-1658.jsonl" \
  --test-size 1658 \
  --output "${scratch_root}/data/tomato-family-manifest.json"

calibration_output="${scratch_root}/data/tomato-open-calibration-8.jsonl"
if [[ ! -f "${calibration_output}" ]]; then
  "${venv_dir}/bin/python" "${repo_dir}/scripts/select_calibration_prompts.py" \
    --input "${scratch_root}/data/tomato-open-train-1000.jsonl" \
    --output "${calibration_output}" \
    --size 8
fi
wc -l "${calibration_output}"
sha256sum "${calibration_output}"

ultrafeedback_output="${scratch_root}/data/ultrafeedback-train-1000.jsonl"
ultrafeedback_targets="${scratch_root}/data/ultrafeedback-targets-1000.json"
ultrafeedback_manifest="${scratch_root}/data/ultrafeedback-manifest-1000.json"
if [[ ! -f "${ultrafeedback_output}" || ! -f "${ultrafeedback_targets}" || ! -f "${ultrafeedback_manifest}" ]]; then
  if [[ -e "${ultrafeedback_output}" || -e "${ultrafeedback_targets}" || -e "${ultrafeedback_manifest}" ]]; then
    echo "partial UltraFeedback artifact set exists; refusing an inconsistent resume" >&2
    exit 1
  fi
  "${venv_dir}/bin/python" "${repo_dir}/scripts/prepare_ultrafeedback.py" \
    --size 1000 \
    --seed 17 \
    --output "${ultrafeedback_output}" \
    --targets "${ultrafeedback_targets}" \
    --manifest "${ultrafeedback_manifest}"
fi
wc -l "${ultrafeedback_output}"
sha256sum "${ultrafeedback_output}" "${ultrafeedback_targets}" "${ultrafeedback_manifest}"
