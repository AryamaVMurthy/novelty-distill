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
score_namespace="${SCORE_NAMESPACE:-teacher-scores}"
samples_per_prompt="${SAMPLES_PER_PROMPT:-8}"
cluster_num_shards="${CLUSTER_NUM_SHARDS:-1}"
cluster_shard_index="${CLUSTER_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
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
if [[ -z "${score_namespace}" || "${score_namespace}" == */* || "${score_namespace}" == *..* ]]; then
  echo "SCORE_NAMESPACE must be a path-safe name" >&2
  exit 2
fi
if ! [[ "${cluster_num_shards}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLUSTER_NUM_SHARDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${samples_per_prompt}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLES_PER_PROMPT must be a positive integer" >&2
  exit 2
fi
if ! [[ "${cluster_shard_index}" =~ ^[0-9]+$ ]] \
  || ((cluster_shard_index >= cluster_num_shards)); then
  echo "CLUSTER_SHARD_INDEX must be within CLUSTER_NUM_SHARDS" >&2
  exit 2
fi
if [[ -n "${calibration_run_id}" && "${cluster_num_shards}" != 1 ]]; then
  echo "calibration clustering does not support partition arrays" >&2
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
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9
mkdir -p "${scratch_root}/locks"
exec 8>"${scratch_root}/locks/venv-inference.lock"
flock -x 8
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/inference.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

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
      --samples-per-prompt "${samples_per_prompt}" \
      --embedding-cache-dir "${scratch_root}/cache/embeddings" \
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
  if [[ "${cluster_num_shards}" -eq 1 ]]; then
    clustered_output="${scratch_root}/data/teacher-generations-${generation_job_id}.jsonl"
  else
    cluster_shard_root="${scratch_root}/data/teacher-generation-shards/${generation_job_id}"
    mkdir -p "${cluster_shard_root}"
    clustered_output="$(printf '%s/part-%05d-of-%05d.jsonl' \
      "${cluster_shard_root}" "${cluster_shard_index}" "${cluster_num_shards}")"
  fi
  "${venv_dir}/bin/python" "${repo_dir}/scripts/cluster_teacher_generations.py" \
    --score-dir "${scratch_root}/${score_namespace}/${generation_job_id}" \
    --output "${clustered_output}" \
    --samples-per-prompt "${samples_per_prompt}" \
    --embedding-cache-dir "${scratch_root}/cache/embeddings" \
    --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml" \
    --num-shards "${cluster_num_shards}" \
    --shard-index "${cluster_shard_index}"

  if [[ "${cluster_num_shards}" -eq 1 ]]; then
    "${venv_dir}/bin/python" "${repo_dir}/scripts/derive_teacher_targets.py" \
      --input "${clustered_output}" \
      --output "${scratch_root}/data/${target_name}" \
      --seed 17
  else
    echo "cluster_partition=${cluster_shard_index}/${cluster_num_shards} output=${clustered_output}"
  fi
fi
