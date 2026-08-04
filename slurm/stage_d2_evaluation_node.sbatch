#!/bin/bash
# Stage the exact D2 evaluation dependencies from node01 onto another Turing node.

#SBATCH --job-name=nd-stage-d2
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

source_node="${SOURCE_NODE:-node01}"
if [[ "$(hostname)" == "${source_node}" ]]; then
  echo "staging target must differ from SOURCE_NODE" >&2
  exit 2
fi

source_root="/scratch/${source_node}/${USER}/novelty-distill"
destination_root="/scratch/${USER}/novelty-distill"
mkdir -p \
  "${destination_root}/cache/huggingface/hub" \
  "${destination_root}/cache" \
  "${destination_root}/checkpoints" \
  "${destination_root}/data" \
  "${destination_root}/evaluation-scores" \
  "${destination_root}/evaluations/tomato" \
  "${destination_root}/logs" \
  "${destination_root}/venvs"

copy_path() {
  local relative="$1"
  local destination_parent
  destination_parent="$(dirname "${destination_root}/${relative}")"
  mkdir -p "${destination_parent}"
  rsync -a --partial --info=stats1 \
    "turing:${source_root}/${relative}" \
    "${destination_parent}/"
}

copy_path data/tomato-open-test-1658.jsonl
copy_path data/teacher-targets-tomato1k-v1.json
copy_path checkpoints/D2-tomato1k-seed17
copy_path evaluation-scores/A1-temporal-k16-seed17000
for evaluation_id in \
  A0-temporal-k4-from-k16 \
  B1-tomato1k-seed17-temporal-k4 \
  B2b-tomato1k-seed17-temporal-k4 \
  C1-best1-tomato1k-seed17-temporal-k4 \
  C2-best1-tomato1k-seed17-temporal-k4 \
  D1-tomato1k-seed17-temporal-k4
do
  copy_path "evaluations/tomato/${evaluation_id}.json"
done
copy_path cache/embeddings
copy_path cache/flashinfer-workspace
copy_path cache/torch-extensions
copy_path cache/triton
copy_path cache/cuda
copy_path cache/xdg
for model_id in Qwen3-4B Qwen3-32B-FP8 Qwen3-Embedding-4B; do
  copy_path "cache/huggingface/hub/models--Qwen--${model_id}"
done
copy_path venvs/inference
copy_path venvs/data

test "$(wc -l < "${destination_root}/data/tomato-open-test-1658.jsonl")" -eq 1658
test "$(find "${destination_root}/evaluation-scores/A1-temporal-k16-seed17000" \
  -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 1658
test -s "${destination_root}/checkpoints/D2-tomato1k-seed17/final/adapter_model.safetensors"
test -x "${destination_root}/venvs/inference/bin/python"
test -x "${destination_root}/venvs/data/bin/python"
echo "d2_stage=complete source=${source_node} target=$(hostname)"
