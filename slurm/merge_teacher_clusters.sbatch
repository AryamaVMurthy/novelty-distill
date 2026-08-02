#!/bin/bash
# Merge all GPU clustering partitions and derive the frozen teacher views.

#SBATCH --job-name=nd-merge-clusters
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

repo_ref="${REPO_REF:-codex/implementation}"
generation_id="${GENERATION_ID:-}"
target_name="${TARGET_NAME:-teacher-targets.json}"
expected_prompts="${EXPECTED_PROMPTS:-}"
cluster_num_shards="${CLUSTER_NUM_SHARDS:-4}"
if [[ -z "${generation_id}" || "${generation_id}" == */* || "${generation_id}" == *..* ]]; then
  echo "GENERATION_ID must be a non-empty path-safe name" >&2
  exit 2
fi
if [[ -z "${target_name}" || "${target_name}" == */* || "${target_name}" == *..* ]]; then
  echo "TARGET_NAME must be a path-safe filename" >&2
  exit 2
fi
for value in "${expected_prompts}" "${cluster_num_shards}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_PROMPTS and CLUSTER_NUM_SHARDS must be positive integers" >&2
    exit 2
  fi
done

scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/inference"
cluster_root="${scratch_root}/data/teacher-generation-shards/${generation_id}"
clustered="${scratch_root}/data/teacher-generations-${generation_id}.jsonl"

export UV_CACHE_DIR="${scratch_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
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

"${venv_dir}/bin/python" "${repo_dir}/scripts/merge_teacher_clusters.py" \
  --input-dir "${cluster_root}" \
  --output "${clustered}" \
  --expected-prompts "${expected_prompts}" \
  --num-shards "${cluster_num_shards}"
"${venv_dir}/bin/python" "${repo_dir}/scripts/derive_teacher_targets.py" \
  --input "${clustered}" \
  --output "${scratch_root}/data/${target_name}" \
  --seed 17
test -s "${clustered}"
test -s "${clustered}.metadata.json"
test -s "${scratch_root}/data/${target_name}"
