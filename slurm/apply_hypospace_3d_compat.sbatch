#!/bin/bash
# Apply the documented NumPy truth-value compatibility patch to pinned HypoSpace 3D.

#SBATCH --job-name=nd-patch-3d
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

official_root="/scratch/${USER}/novelty-distill/official/hypospace"
benchmark_file="${official_root}/3d/run_3d_benchmark.py"
test -f "${benchmark_file}"
test "$(git -C "${official_root}" rev-parse HEAD)" = \
  "c69e9318577b34b5b896996571aefd4ba6053f58"

python3 - "${benchmark_file}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """        for layer in layers:\n            if layer:  # Non-empty layer\n                max_rows = max(max_rows, len(layer))\n                for row in layer:\n                    if row:  # Non-empty row\n                        max_cols = max(max_cols, len(row))\n"""
new = """        for layer in layers:\n            # NumPy arrays have no scalar truth value; use length for both\n            # list-backed and array-backed layers from model outputs.\n            if layer is not None and len(layer) > 0:  # Non-empty layer\n                max_rows = max(max_rows, len(layer))\n                for row in layer:\n                    if row is not None and len(row) > 0:  # Non-empty row\n                        max_cols = max(max_cols, len(row))\n"""
if new in text:
    print("compatibility_patch=already_present")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("compatibility_patch=applied")
else:
    raise SystemExit("expected pinned HypoSpace 3D source text was not found")
PY

grep -n -A8 -B2 "NumPy arrays have no scalar truth value" "${benchmark_file}"
echo "official_head=$(git -C "${official_root}" rev-parse HEAD) target=$(hostname)"
