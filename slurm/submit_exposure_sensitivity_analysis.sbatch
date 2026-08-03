#!/bin/bash
# Resolve the dynamically submitted primary analysis and join it to the 4x study.

#SBATCH --job-name=nd-submit-exposure-analysis
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

primary_controller_job_id="${PRIMARY_CONTROLLER_JOB_ID:?set primary evaluation controller job ID}"
exposure_evaluation_job_ids="${EXPOSURE_EVALUATION_JOB_IDS:?set six exposure evaluation job IDs}"
exposure_taste_job_ids="${EXPOSURE_TASTE_JOB_IDS:?set six exposure taste job IDs}"
expected_exposure_jobs="${EXPECTED_EXPOSURE_JOBS:-6}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
controller_log="${scratch_root}/logs/slurm-nd-submit-evals-${primary_controller_job_id}.out"
submission_dir="${scratch_root}/submissions"
submission_path="${submission_dir}/exposure-sensitivity-analysis.json"

if ! [[ "${primary_controller_job_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "primary controller job ID must be a positive integer" >&2
  exit 2
fi
if ! [[ "${expected_exposure_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "expected exposure job count must be a positive integer" >&2
  exit 2
fi

normalize_job_ids() {
  local raw_ids="$1"
  local expected_count="$2"
  local normalized="${raw_ids//,/:}"
  local values=()
  local job_id
  IFS=: read -r -a values <<< "${normalized}"
  if [[ "${#values[@]}" -ne "${expected_count}" ]]; then
    echo "expected ${expected_count} job IDs, got ${#values[@]}" >&2
    return 2
  fi
  for job_id in "${values[@]}"; do
    if ! [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
      echo "invalid Slurm job ID: ${job_id}" >&2
      return 2
    fi
  done
  (IFS=:; printf '%s' "${values[*]}")
}

evaluation_dependencies="$(
  normalize_job_ids "${exposure_evaluation_job_ids}" "${expected_exposure_jobs}"
)"
taste_dependencies="$(
  normalize_job_ids "${exposure_taste_job_ids}" "${expected_exposure_jobs}"
)"

test -s "${controller_log}"
primary_analysis_job_id="$(
  python3 - "${controller_log}" <<'PY'
import json
import pathlib
import sys

lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
for line in reversed(lines):
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    analysis = record.get("analysis")
    if isinstance(analysis, str) and analysis.isdigit() and int(analysis) > 0:
        print(analysis)
        break
else:
    raise SystemExit("primary controller log has no valid analysis job ID")
PY
)"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" \
  "${repo_dir}" "${REPO_REF:-codex/implementation}"
flock -u 9

all_dependencies="${primary_analysis_job_id}:${evaluation_dependencies}:${taste_dependencies}"
cd "${repo_dir}"
analysis_job_id="$(
  sbatch --parsable \
    --dependency="afterok:${all_dependencies}" \
    --output="${scratch_root}/logs/slurm-%x-%j.out" \
    --error="${scratch_root}/logs/slurm-%x-%j.err" \
    slurm/analyze_exposure_sensitivity.sbatch
)"
analysis_job_id="${analysis_job_id%%;*}"
if ! [[ "${analysis_job_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "could not parse exposure analysis job ID: ${analysis_job_id}" >&2
  exit 1
fi

mkdir -p "${submission_dir}"
PRIMARY_CONTROLLER_JOB_ID="${primary_controller_job_id}" \
PRIMARY_ANALYSIS_JOB_ID="${primary_analysis_job_id}" \
EXPOSURE_EVALUATION_JOB_IDS="${evaluation_dependencies}" \
EXPOSURE_TASTE_JOB_IDS="${taste_dependencies}" \
EXPOSURE_ANALYSIS_JOB_ID="${analysis_job_id}" \
SUBMISSION_PATH="${submission_path}" \
python3 - <<'PY'
import json
import os
import pathlib
import tempfile

path = pathlib.Path(os.environ["SUBMISSION_PATH"])
record = {
    "study": "tomato1k-exposure4x",
    "primary_controller": os.environ["PRIMARY_CONTROLLER_JOB_ID"],
    "primary_analysis": os.environ["PRIMARY_ANALYSIS_JOB_ID"],
    "exposure_evaluations": os.environ["EXPOSURE_EVALUATION_JOB_IDS"].split(":"),
    "exposure_taste": os.environ["EXPOSURE_TASTE_JOB_IDS"].split(":"),
    "analysis": os.environ["EXPOSURE_ANALYSIS_JOB_ID"],
}
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
) as handle:
    json.dump(record, handle, sort_keys=True)
    handle.write("\n")
    temporary_path = handle.name
os.replace(temporary_path, path)
print(json.dumps(record, sort_keys=True))
PY
