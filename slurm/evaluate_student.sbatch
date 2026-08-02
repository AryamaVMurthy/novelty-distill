#!/bin/bash
#SBATCH --job-name=nd-evaluate-student
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

module load u22/cuda/12.4

teacher_score_id="${TEACHER_SCORE_ID:-}"
student_score_id="${STUDENT_SCORE_ID:-}"
score_namespace="${SCORE_NAMESPACE:-evaluation-scores}"
target_name="${TARGET_NAME:-teacher-targets-tomato1k-v1.json}"
output_name="${OUTPUT_NAME:-}"
teacher_samples_per_prompt="${TEACHER_SAMPLES_PER_PROMPT:-16}"
student_samples_per_prompt="${STUDENT_SAMPLES_PER_PROMPT:-16}"
for value in "${teacher_score_id}" "${student_score_id}" "${score_namespace}" "${target_name}" "${output_name}"; do
  if [[ -z "${value}" || "${value}" == */* || "${value}" == *..* ]]; then
    echo "score IDs, namespace, target, and output names must be non-empty path-safe names" >&2
    exit 2
  fi
done
for value in "${teacher_samples_per_prompt}" "${student_samples_per_prompt}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "teacher and student samples per prompt must be positive integers" >&2
    exit 2
  fi
done

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

mkdir -p "${scratch_root}/tmp" "${scratch_root}/evaluations/tomato"
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
flock -u 9
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/inference.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

"${venv_dir}/bin/python" "${repo_dir}/scripts/evaluate_student_generations.py" \
  --teacher-score-dir "${scratch_root}/${score_namespace}/${teacher_score_id}" \
  --student-score-dir "${scratch_root}/${score_namespace}/${student_score_id}" \
  --training-targets "${scratch_root}/data/${target_name}" \
  --annotation-config "${repo_dir}/configs/evaluation/teacher_annotation.yaml" \
  --embedding-cache-dir "${scratch_root}/cache/embeddings" \
  --output "${scratch_root}/evaluations/tomato/${output_name}.json" \
  --teacher-samples-per-prompt "${teacher_samples_per_prompt}" \
  --student-samples-per-prompt "${student_samples_per_prompt}"

test -s "${scratch_root}/evaluations/tomato/${output_name}.json"
