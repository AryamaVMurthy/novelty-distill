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

calibration_run_id="${CALIBRATION_RUN_ID:-}"
generation_job_id="${GENERATION_JOB_ID:-}"
target_name="${TARGET_NAME:-teacher-targets.json}"
if [[ -z "${calibration_run_id}" && -z "${generation_job_id}" ]]; then
  echo "set GENERATION_JOB_ID or CALIBRATION_RUN_ID" >&2
  exit 2
fi
if [[ -n "${calibration_run_id}" && ("${calibration_run_id}" == */* || "${calibration_run_id}" == *..*) ]]; then
  echo "CALIBRATION_RUN_ID must be a path-safe name" >&2
  exit 2
fi
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

if [[ -n "${calibration_run_id}" ]]; then
  score_root="${scratch_root}/teacher-calibration-scores/${calibration_run_id}"
  cluster_root="${scratch_root}/teacher-calibration-clusters/${calibration_run_id}"
  target_root="${scratch_root}/data/teacher-calibration-targets/${calibration_run_id}"
  mkdir -p "${cluster_root}" "${target_root}"
  conditions=0
  for score_dir in "${score_root}"/*; do
    [[ -d "${score_dir}" ]] || continue
    condition="$(basename "${score_dir}")"
    clustered="${cluster_root}/${condition}.jsonl"
    "${venv_dir}/bin/python" "${repo_dir}/scripts/cluster_teacher_generations.py" \
      --score-dir "${score_dir}" \
      --output "${clustered}" \
      --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml"
    "${venv_dir}/bin/python" "${repo_dir}/scripts/derive_teacher_targets.py" \
      --input "${clustered}" \
      --output "${target_root}/${condition}.json" \
      --seed 17
    conditions=$((conditions + 1))
  done
  test "${conditions}" -gt 0
  echo "clustered_calibration_conditions=${conditions}"
  analysis_root="${scratch_root}/evaluations/teacher-calibration/${calibration_run_id}"
  mkdir -p "${analysis_root}"
  "${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_teacher_calibration.py" \
    --generation-root "${scratch_root}/generations/teacher-calibration/${calibration_run_id}" \
    --cluster-root "${cluster_root}" \
    --output "${analysis_root}/analysis.json" \
    --markdown "${analysis_root}/findings.md"
  test -s "${analysis_root}/analysis.json"
  test -s "${analysis_root}/findings.md"
else
  "${venv_dir}/bin/python" "${repo_dir}/scripts/cluster_teacher_generations.py" \
    --score-dir "${scratch_root}/teacher-scores/${generation_job_id}" \
    --output "${scratch_root}/data/teacher-generations-${generation_job_id}.jsonl" \
    --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml"

  "${venv_dir}/bin/python" "${repo_dir}/scripts/derive_teacher_targets.py" \
    --input "${scratch_root}/data/teacher-generations-${generation_job_id}.jsonl" \
    --output "${scratch_root}/data/${target_name}" \
    --seed 17
fi
