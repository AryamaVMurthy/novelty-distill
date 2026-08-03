#!/bin/bash
# Analyze a balanced 5k/20k promoted matrix across its declared training seeds.

#SBATCH --job-name=nd-analyze-promoted
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

model_profile="${MODEL_PROFILE:-main}"
train_size="${TRAIN_SIZE:?set promoted TRAIN_SIZE}"
promoted_methods="${PROMOTED_METHODS:?set comma-separated promoted baseline IDs}"
train_seeds="${TRAIN_SEEDS:?set comma-separated promoted training seeds}"
case "${model_profile}:${train_size}" in
  main:5000|main:20000|replication:1000|replication:5000|replication:20000) ;;
  *) echo "unsupported promoted model profile and train size" >&2; exit 2 ;;
esac

IFS=',' read -r -a baseline_ids <<<"${promoted_methods}"
IFS=',' read -r -a seeds <<<"${train_seeds}"
declare -A seen_baseline=()
declare -A seen_seed=()
for baseline_id in "${baseline_ids[@]}"; do
  if ! [[ "${baseline_id}" =~ ^[A-Z][A-Za-z0-9-]*$ ]] || \
    [[ -n "${seen_baseline[$baseline_id]:-}" ]]
  then
    echo "PROMOTED_METHODS contains an invalid or duplicate baseline ID" >&2
    exit 2
  fi
  seen_baseline[$baseline_id]=1
done
for seed in "${seeds[@]}"; do
  if [[ "${seed}" != 17 && "${seed}" != 29 && "${seed}" != 43 ]] || \
    [[ -n "${seen_seed[$seed]:-}" ]]
  then
    echo "TRAIN_SEEDS must contain unique values from 17,29,43" >&2
    exit 2
  fi
  seen_seed[$seed]=1
done

repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
venv_dir="${scratch_root}/venvs/data"
evaluation_root="${scratch_root}/evaluations/tomato"
taste_root="${scratch_root}/research-taste"
if [[ "${model_profile}" == main ]]; then
  run_infix="tomato${train_size}"
  default_a0_taste="A0-temporal-k16-seed17000"
  default_a1_taste="A1-temporal-k16-seed17000"
  default_a3_taste="A3-temporal-k1"
else
  run_infix="qwen1p7b-tomato${train_size}"
  default_a0_taste="A0-${run_infix}-temporal-k16"
  default_a1_taste="A1-${run_infix}-temporal-k16"
  default_a3_taste="A3-temporal-k1"
fi
output_dir="${evaluation_root}/analysis-${run_infix}"
control_a0_eval_id="${CONTROL_A0_EVAL_ID:-A0-${run_infix}-temporal-k16}"
control_a1_eval_id="${CONTROL_A1_EVAL_ID:-A1-${run_infix}-temporal-k16}"
control_a3_eval_id="${CONTROL_A3_EVAL_ID:-A3-${run_infix}-temporal-k1}"
control_a0_taste_id="${CONTROL_A0_TASTE_ID:-${default_a0_taste}}"
control_a1_taste_id="${CONTROL_A1_TASTE_ID:-${default_a1_taste}}"
control_a3_taste_id="${CONTROL_A3_TASTE_ID:-${default_a3_taste}}"

exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9

export UV_CACHE_DIR="${scratch_root}/cache/uv-analysis"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_LINK_MODE=copy
mkdir -p "${scratch_root}/locks" "${output_dir}"
exec 8>"${scratch_root}/locks/venv-data.lock"
flock -x 8
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/data.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

collector_inputs=(
  --control "A0=${evaluation_root}/${control_a0_eval_id}.json"
  --control "A1=${evaluation_root}/${control_a1_eval_id}.json"
  --control "A3=${evaluation_root}/${control_a3_eval_id}.json"
)
taste_inputs=(
  --input "A0=${taste_root}/${control_a0_taste_id}"
  --input "A1=${taste_root}/${control_a1_taste_id}"
  --input "A3=${taste_root}/${control_a3_taste_id}"
)
wandb_inputs=(
  --input "${evaluation_root}/${control_a0_eval_id}.json"
  --input "${evaluation_root}/${control_a1_eval_id}.json"
  --input "${evaluation_root}/${control_a3_eval_id}.json"
)
for control_id in "${control_a0_eval_id}" "${control_a1_eval_id}" "${control_a3_eval_id}"; do
  test -s "${evaluation_root}/${control_id}.json"
done
for control_taste in "${control_a0_taste_id}" "${control_a1_taste_id}" "${control_a3_taste_id}"; do
  test -d "${taste_root}/${control_taste}"
done

for seed in "${seeds[@]}"; do
  audit_args=(
    --registry "${repo_dir}/configs/baselines.yaml"
    --output "${output_dir}/training-matrix-audit-seed${seed}.json"
  )
  for baseline_id in "${baseline_ids[@]}"; do
    run_name="${baseline_id}-${run_infix}-seed${seed}"
    eval_id="${run_name}-temporal-k16"
    if [[ "${baseline_id}" == C3 ]]; then
      metadata_path="${scratch_root}/checkpoints/${run_name}-l896-4gpu/run_metadata.json"
    else
      metadata_path="${scratch_root}/checkpoints/${run_name}/run_metadata.json"
    fi
    test -s "${metadata_path}"
    test -s "${evaluation_root}/${eval_id}.json"
    test -d "${taste_root}/${eval_id}"
    audit_args+=(
      --baseline-id "${baseline_id}"
      --input "${baseline_id}=${metadata_path}"
    )
    collector_inputs+=(--input "${baseline_id}:${seed}=${evaluation_root}/${eval_id}.json")
    taste_inputs+=(--input "${baseline_id}-seed${seed}=${taste_root}/${eval_id}")
    wandb_inputs+=(--input "${metadata_path}" --input "${evaluation_root}/${eval_id}.json")
  done
  "${venv_dir}/bin/python" "${repo_dir}/scripts/audit_training_matrix.py" \
    "${audit_args[@]}"
  wandb_inputs+=(--input "${output_dir}/training-matrix-audit-seed${seed}.json")
done

matrix_path="${output_dir}/prompt-metrics.jsonl"
threshold_matrix_path="${output_dir}/threshold-prompt-metrics.jsonl"
"${venv_dir}/bin/python" "${repo_dir}/scripts/collect_seeded_evaluation_metrics.py" \
  "${collector_inputs[@]}" \
  --seeds "${train_seeds}" \
  --output "${matrix_path}" \
  --threshold-output "${threshold_matrix_path}" \
  --manifest "${output_dir}/seeded-collection-manifest.json"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_contrast_matrix.py" \
  --input "${matrix_path}" \
  --threshold-input "${threshold_matrix_path}" \
  --config "${repo_dir}/configs/evaluation/primary_contrasts.yaml" \
  --filter-absent-contrasts \
  --output "${output_dir}/primary-contrasts.json" \
  --markdown "${output_dir}/findings.md"
"${venv_dir}/bin/python" "${repo_dir}/scripts/analyze_research_taste.py" \
  "${taste_inputs[@]}" \
  --human-method A3 \
  --teacher-method A1 \
  --config "${repo_dir}/configs/evaluation/research_taste.yaml" \
  --output "${output_dir}/research-taste-secondary.json" \
  --markdown "${output_dir}/research-taste-findings.md"

wandb_inputs+=(
  --input "${output_dir}/seeded-collection-manifest.json"
  --input "${output_dir}/primary-contrasts.json"
  --input "${output_dir}/research-taste-secondary.json"
)
"${venv_dir}/bin/python" "${repo_dir}/scripts/export_wandb_snapshot.py" \
  "${wandb_inputs[@]}" \
  --manifest "${output_dir}/wandb-export-manifest.json" \
  --run-dir "${scratch_root}/wandb"

test -s "${matrix_path}"
test -s "${threshold_matrix_path}"
test -s "${output_dir}/seeded-collection-manifest.json"
test -s "${output_dir}/primary-contrasts.json"
test -s "${output_dir}/findings.md"
test -s "${output_dir}/research-taste-secondary.json"
test -s "${output_dir}/research-taste-findings.md"
test -s "${output_dir}/wandb-export-manifest.json"
