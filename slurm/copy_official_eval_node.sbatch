#!/bin/bash
# Copy completed official component artifacts between node-local scratch roots.

#SBATCH --job-name=nd-copy-official
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

source_node="${SOURCE_NODE:?set SOURCE_NODE}"
eval_id="${EVAL_ID:?set EVAL_ID}"
if [[ ! "${source_node}" =~ ^node[0-9]+$ ]]; then
  echo "SOURCE_NODE must be a node name" >&2
  exit 2
fi
if [[ -z "${eval_id}" || "${eval_id}" == */* || "${eval_id}" == *..* ]]; then
  echo "EVAL_ID must be path-safe" >&2
  exit 2
fi

source_root="/scratch/${source_node}/${USER}/novelty-distill/evaluations/official/${eval_id}"
destination_root="/scratch/${USER}/novelty-distill/evaluations/official/${eval_id}"
mkdir -p "${destination_root}"
rsync -a --partial --info=stats1 \
  "turing:${source_root}/" "${destination_root}/"
echo "official_eval_copy=complete source=${source_node} target=$(hostname) eval_id=${eval_id}"
