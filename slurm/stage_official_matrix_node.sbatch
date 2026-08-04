#!/bin/bash
# Stage the compact-model official-benchmark dependencies onto one Turing node.

#SBATCH --job-name=nd-stage-official
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

source_node="${SOURCE_NODE:-node01}"
if [[ "$(hostname)" == "${source_node}" ]]; then
  echo "staging target must differ from SOURCE_NODE" >&2
  exit 2
fi

source_root="/scratch/${source_node}/${USER}/novelty-distill"
destination_root="/scratch/${USER}/novelty-distill"
mkdir -p "${destination_root}"

copy_path() {
  local relative="$1"
  local destination_parent
  destination_parent="$(dirname "${destination_root}/${relative}")"
  mkdir -p "${destination_parent}"
  rsync -a --partial --info=stats1 \
    "turing:${source_root}/${relative}" \
    "${destination_parent}/"
}

copy_path official/inspect_evals
copy_path official/hypospace
copy_path venvs/inference
copy_path venvs/hypospace
copy_path venvs/noveltybench
copy_path cache/uv
copy_path cache/huggingface/hub/models--Qwen--Qwen3-4B
copy_path cache/huggingface/hub/models--yimingzhang--deberta-v3-large-generation-similarity
copy_path cache/huggingface/hub/models--microsoft--deberta-v3-large

for baseline_id in \
  B1 \
  B2b \
  C1-best1 \
  C2-best1 \
  D1 \
  D2
do
  copy_path "checkpoints/${baseline_id}-tomato1k-seed17/final"
done

mkdir -p \
  "${destination_root}/cache/cuda" \
  "${destination_root}/cache/flashinfer-workspace" \
  "${destination_root}/cache/torch-extensions" \
  "${destination_root}/cache/triton" \
  "${destination_root}/cache/tvm-ffi" \
  "${destination_root}/cache/uv" \
  "${destination_root}/cache/xdg" \
  "${destination_root}/evaluations/official" \
  "${destination_root}/locks" \
  "${destination_root}/logs" \
  "${destination_root}/tmp"

test "$(git -C "${destination_root}/official/inspect_evals" rev-parse HEAD)" = \
  "6a35510e530f236fd1dbcd9df888f01937c8494a"
test "$(git -C "${destination_root}/official/hypospace" rev-parse HEAD)" = \
  "c69e9318577b34b5b896996571aefd4ba6053f58"
test -x "${destination_root}/venvs/inference/bin/python"
test -x "${destination_root}/venvs/hypospace/bin/python"
test -x "${destination_root}/venvs/noveltybench/bin/python"
test -s \
  "${destination_root}/official/hypospace/causal/datasets/node03/n3_all_observations.json"
test -s "${destination_root}/official/hypospace/3d/datasets/3d_grid3_h3.json"
test -s "${destination_root}/official/hypospace/boolean/datasets/boolean_2var.json"
for baseline_id in B1 B2b C1-best1 C2-best1 D1 D2; do
  test -s \
    "${destination_root}/checkpoints/${baseline_id}-tomato1k-seed17/final/adapter_model.safetensors"
done

echo "official_matrix_stage=complete source=${source_node} target=$(hostname)"
