#!/bin/bash
#SBATCH --job-name=nd-train-smoke
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

module load u22/cuda/12.4

repo_ref="${REPO_REF:-codex/implementation}"
scratch_root="/scratch/${USER}/novelty-distill"
repo_dir="${scratch_root}/repo"
training_backend="${TRAINING_BACKEND:-trl}"
matrix_mode="${BASELINE_MATRIX:-0}"
if [[ "${matrix_mode}" == "tomato1k" ]]; then
  run_suffix="${RUN_SUFFIX:-tomato1k-seed17}"
elif [[ "${matrix_mode}" == "tomato_scale" ]]; then
  scale_train_size="${TRAIN_SIZE:?tomato_scale requires TRAIN_SIZE}"
  scale_seed="${TRAIN_SEED:?tomato_scale requires TRAIN_SEED}"
  model_profile="${MODEL_PROFILE:-main}"
  if [[ "${model_profile}" == main ]]; then
    valid_scale_sizes="5000 20000"
  elif [[ "${model_profile}" == replication ]]; then
    valid_scale_sizes="1000 5000 20000"
  else
    echo "MODEL_PROFILE must be main or replication" >&2
    exit 2
  fi
  if [[ " ${valid_scale_sizes} " != *" ${scale_train_size} "* ]]; then
    echo "TRAIN_SIZE is invalid for MODEL_PROFILE=${model_profile}" >&2
    exit 2
  fi
  if [[ "${scale_seed}" != 17 && "${scale_seed}" != 29 && "${scale_seed}" != 43 ]]; then
    echo "TRAIN_SEED must be 17, 29, or 43" >&2
    exit 2
  fi
  if [[ "${model_profile}" == replication ]]; then
    default_run_suffix="qwen1p7b-tomato${scale_train_size}-seed${scale_seed}"
  else
    default_run_suffix="tomato${scale_train_size}-seed${scale_seed}"
  fi
  run_suffix="${RUN_SUFFIX:-${default_run_suffix}}"
elif [[ "${matrix_mode}" == "ultrafeedback1k" ]]; then
  run_suffix="${RUN_SUFFIX:-ultrafeedback1k-seed17}"
else
  run_suffix="${RUN_SUFFIX:-smoke}"
fi
if [[ -z "${run_suffix}" || "${run_suffix}" == */* || "${run_suffix}" == *..* ]]; then
  echo "RUN_SUFFIX must be a path-safe non-empty name" >&2
  exit 2
fi
if [[ "${matrix_mode}" == 1 ]]; then
  matrix_ids=(
    B2a B2b B2c B3
    C1-human C1-best1 C1-diverse4 C2-human C2-best1 C2-diverse4 D2 D3
    E3 E4 B4 C3
  )
  matrix_backends=(
    trl trl trl trl
    trl trl trl trl trl trl trl trl
    opsd opsd gem distillm
  )
  matrix_configs=(
    configs/training/sft_smoke.yaml configs/training/sft_smoke.yaml
    configs/training/sft_smoke.yaml configs/training/sft_smoke.yaml
    configs/training/gkd_smoke.yaml configs/training/gkd_smoke.yaml
    configs/training/gkd_smoke.yaml configs/training/gkd_smoke.yaml
    configs/training/gkd_smoke.yaml configs/training/gkd_smoke.yaml
    configs/training/gkd_smoke.yaml configs/training/gkd_smoke.yaml
    configs/training/opsd_smoke.yaml configs/training/opsd_smoke.yaml
    configs/training/gem_smoke.yaml configs/training/distillm_smoke.yaml
  )
  matrix_index="${SLURM_ARRAY_TASK_ID:?BASELINE_MATRIX requires a Slurm array task}"
  if ((matrix_index < 0 || matrix_index >= ${#matrix_ids[@]})); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#matrix_ids[@]} - 1))" >&2
    exit 1
  fi
  BASELINE_ID="${matrix_ids[$matrix_index]}"
  training_backend="${matrix_backends[$matrix_index]}"
  RUN_CONFIG="${matrix_configs[$matrix_index]}"
  echo "matrix_index=${matrix_index} baseline_id=${BASELINE_ID} backend=${training_backend}"
elif [[ "${matrix_mode}" == "ultrafeedback1k" ]]; then
  matrix_ids=(B1 B2a B2b B3 D1)
  matrix_backends=(trl trl trl trl trl)
  matrix_configs=(
    configs/training/sft_ultrafeedback1k.yaml
    configs/training/sft_ultrafeedback1k.yaml
    configs/training/sft_ultrafeedback1k.yaml
    configs/training/sft_ultrafeedback1k.yaml
    configs/training/gkd_ultrafeedback1k.yaml
  )
  matrix_index="${SLURM_ARRAY_TASK_ID:?ultrafeedback1k matrix requires a Slurm array task}"
  if ((matrix_index < 0 || matrix_index >= ${#matrix_ids[@]})); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#matrix_ids[@]} - 1))" >&2
    exit 1
  fi
  BASELINE_ID="${matrix_ids[$matrix_index]}"
  training_backend="${matrix_backends[$matrix_index]}"
  RUN_CONFIG="${matrix_configs[$matrix_index]}"
  echo "matrix=ultrafeedback1k index=${matrix_index} baseline_id=${BASELINE_ID} backend=${training_backend}"
elif [[ "${matrix_mode}" == "tomato1k" ]]; then
  matrix_ids=(
    B1 B2a B2b B2c B3 B4
    C1-human C1-best1 C1-diverse4 C2-human C2-best1 C2-diverse4 C3
    D1 D2 D3 E2 E3 E4
  )
  matrix_backends=(
    trl trl trl trl trl gem
    trl trl trl trl trl trl distillm
    trl trl trl opsd opsd opsd
  )
  matrix_configs=(
    configs/training/sft_tomato1k.yaml configs/training/sft_tomato1k.yaml
    configs/training/sft_tomato1k.yaml configs/training/sft_tomato1k.yaml
    configs/training/sft_tomato1k.yaml configs/training/gem_tomato1k.yaml
    configs/training/gkd_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/gkd_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/gkd_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/distillm_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/gkd_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/opsd_tomato1k.yaml configs/training/opsd_tomato1k.yaml
    configs/training/opsd_tomato1k.yaml
  )
  matrix_index="${SLURM_ARRAY_TASK_ID:?tomato1k matrix requires a Slurm array task}"
  if ((matrix_index < 0 || matrix_index >= ${#matrix_ids[@]})); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#matrix_ids[@]} - 1))" >&2
    exit 1
  fi
  BASELINE_ID="${matrix_ids[$matrix_index]}"
  training_backend="${matrix_backends[$matrix_index]}"
  RUN_CONFIG="${matrix_configs[$matrix_index]}"
  echo "matrix=tomato1k index=${matrix_index} baseline_id=${BASELINE_ID} backend=${training_backend}"
elif [[ "${matrix_mode}" == "tomato_scale" ]]; then
  matrix_ids=(
    B1 B2a B2b B2c B3 B4
    C1-human C1-best1 C1-diverse4 C2-human C2-best1 C2-diverse4 C3
    D1 D2 D3 E2 E3 E4
  )
  matrix_backends=(
    trl trl trl trl trl gem
    trl trl trl trl trl trl distillm
    trl trl trl opsd opsd opsd
  )
  matrix_configs=(
    configs/training/sft_tomato1k.yaml configs/training/sft_tomato1k.yaml
    configs/training/sft_tomato1k.yaml configs/training/sft_tomato1k.yaml
    configs/training/sft_tomato1k.yaml configs/training/gem_tomato1k.yaml
    configs/training/gkd_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/gkd_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/gkd_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/distillm_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/gkd_tomato1k.yaml configs/training/gkd_tomato1k.yaml
    configs/training/opsd_tomato1k.yaml configs/training/opsd_tomato1k.yaml
    configs/training/opsd_tomato1k.yaml
  )
  matrix_index="${SLURM_ARRAY_TASK_ID:?tomato_scale matrix requires a Slurm array task}"
  if ((matrix_index < 0 || matrix_index >= ${#matrix_ids[@]})); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#matrix_ids[@]} - 1))" >&2
    exit 1
  fi
  BASELINE_ID="${matrix_ids[$matrix_index]}"
  training_backend="${matrix_backends[$matrix_index]}"
  RUN_CONFIG="${matrix_configs[$matrix_index]}"
  echo "matrix=tomato_scale size=${scale_train_size} seed=${scale_seed} index=${matrix_index} baseline_id=${BASELINE_ID} backend=${training_backend}"
fi
if [[ "${training_backend}" == "opsd" ]]; then
  run_config="${RUN_CONFIG:-configs/training/opsd_smoke.yaml}"
  venv_dir="${scratch_root}/venvs/training"
  environment_lock="${repo_dir}/environments/training.lock"
elif [[ "${training_backend}" == "trl" ]]; then
  run_config="${RUN_CONFIG:-configs/training/sft_smoke.yaml}"
  venv_dir="${scratch_root}/venvs/trl"
  environment_lock="${repo_dir}/environments/trl.lock"
elif [[ "${training_backend}" == "gem" ]]; then
  run_config="${RUN_CONFIG:-configs/training/gem_smoke.yaml}"
  venv_dir="${scratch_root}/venvs/gem"
  environment_lock="${repo_dir}/environments/gem.lock"
elif [[ "${training_backend}" == "distillm" ]]; then
  run_config="${RUN_CONFIG:-configs/training/distillm_smoke.yaml}"
  venv_dir="${scratch_root}/venvs/distillm"
  environment_lock="${repo_dir}/environments/distillm.lock"
else
  echo "unsupported TRAINING_BACKEND=${training_backend}" >&2
  exit 1
fi
config_path="${repo_dir}/${run_config}"
monitor_file="${scratch_root}/logs/gpu-${SLURM_JOB_ID}.csv"
monitor_pid=""

export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${scratch_root}/cache/uv-training"
export UV_PYTHON_INSTALL_DIR="${scratch_root}/python"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
export HF_HOME="${scratch_root}/cache/huggingface"
export TORCH_HOME="${scratch_root}/cache/torch"
export XDG_CACHE_HOME="${scratch_root}/cache/xdg"
export TORCH_EXTENSIONS_DIR="${scratch_root}/cache/torch-extensions"
export TRITON_CACHE_DIR="${scratch_root}/cache/triton"
export CUDA_CACHE_PATH="${scratch_root}/cache/cuda"
export TMPDIR="${scratch_root}/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cleanup() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}"
    wait "${monitor_pid}" || true
  fi
}
trap cleanup EXIT

mkdir -p \
  "${scratch_root}/logs" \
  "${scratch_root}/locks" \
  "${scratch_root}/cache/torch-extensions" \
  "${scratch_root}/cache/triton" \
  "${scratch_root}/cache/cuda" \
  "${scratch_root}/cache/xdg" \
  "${scratch_root}/tmp"
exec 9>"${repo_dir}/.git/novelty-distill-sync.lock"
flock -x 9
"${repo_dir}/scripts/sync_repository.sh" "${repo_dir}" "${repo_ref}"
flock -u 9
export NOVELTY_GIT_COMMIT
NOVELTY_GIT_COMMIT="$(git -C "${repo_dir}" rev-parse HEAD)"

exec 8>"${scratch_root}/locks/venv-${training_backend}.lock"
flock -x 8
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  uv venv --python 3.10 "${venv_dir}"
fi
"${venv_dir}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
uv pip sync --python "${venv_dir}/bin/python" "${environment_lock}"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"
flock -u 8

if [[ "${matrix_mode}" == "tomato_scale" ]]; then
  rendered_config="${scratch_root}/tmp/config-${BASELINE_ID}-${run_suffix}.yaml"
  "${venv_dir}/bin/python" "${repo_dir}/scripts/render_scale_training_config.py" \
    --base "${config_path}" \
    --output "${rendered_config}" \
    --backend "${training_backend}" \
    --baseline-id "${BASELINE_ID}" \
    --train-size "${scale_train_size}" \
    --seed "${scale_seed}"
    --model-profile "${model_profile}"
  config_path="${rendered_config}"
fi

if [[ -n "${BASELINE_ID:-}" ]] && [[ "${training_backend}" == "trl" || "${training_backend}" == "opsd" ]]; then
  metadata_path="${scratch_root}/checkpoints/${BASELINE_ID}-${run_suffix}/run_metadata.json"
  max_steps="$(${venv_dir}/bin/python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["max_steps"])' "${config_path}")"
  set +e
  "${venv_dir}/bin/python" "${repo_dir}/scripts/check_training_status.py" \
    --metadata "${metadata_path}" \
    --baseline-id "${BASELINE_ID}" \
    --max-steps "${max_steps}"
  training_status=$?
  set -e
  if [[ "${training_status}" -eq 0 ]]; then
    echo "stage=already_complete output=${metadata_path}"
    exit 0
  fi
  if [[ "${training_status}" -ne 3 ]]; then
    echo "training preflight failed with status ${training_status}" >&2
    exit "${training_status}"
  fi
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "git_commit=${NOVELTY_GIT_COMMIT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "start_time=$(date --iso-8601=seconds)"
nvidia-smi
if [[ "${training_backend}" == "gem" ]]; then
  "${venv_dir}/bin/python" -c 'import datasets, deepspeed, torch, transformers; print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "datasets": datasets.__version__, "deepspeed": deepspeed.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0)})'
elif [[ "${training_backend}" == "distillm" ]]; then
  "${venv_dir}/bin/python" -c 'import deepspeed, peft, torch, transformers; print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "deepspeed": deepspeed.__version__, "peft": peft.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0)})'
else
  "${venv_dir}/bin/python" -c 'import torch, transformers, trl, peft; print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0)})'
fi

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 5 >"${monitor_file}" &
monitor_pid="$!"

if [[ "${training_backend}" == "opsd" ]]; then
  opsd_args=(
    --config "${config_path}"
    --scratch-root "${scratch_root}"
    --registry "${repo_dir}/configs/baselines.yaml"
    --manifest "${repo_dir}/third_party/manifest.yaml"
    --official-root "${scratch_root}/official"
  )
  if [[ -n "${BASELINE_ID:-}" ]]; then
    opsd_args+=(--baseline-id "${BASELINE_ID}" --run-suffix "${run_suffix}")
  fi
  "${venv_dir}/bin/python" "${repo_dir}/scripts/run_opsd_baseline.py" "${opsd_args[@]}"
elif [[ "${training_backend}" == "trl" ]]; then
  trl_args=(
    --config "${config_path}"
    --scratch-root "${scratch_root}"
    --registry "${repo_dir}/configs/baselines.yaml"
  )
  if [[ -n "${BASELINE_ID:-}" ]]; then
    trl_args+=(--baseline-id "${BASELINE_ID}" --run-suffix "${run_suffix}")
  fi
  "${venv_dir}/bin/python" "${repo_dir}/scripts/run_trl_baseline.py" "${trl_args[@]}"
elif [[ "${training_backend}" == "gem" ]]; then
  "${venv_dir}/bin/python" "${repo_dir}/scripts/run_gem_baseline.py" \
    --config "${config_path}" \
    --scratch-root "${scratch_root}" \
    --registry "${repo_dir}/configs/baselines.yaml" \
    --manifest "${repo_dir}/third_party/manifest.yaml" \
    --official-root "${scratch_root}/official"
else
  "${venv_dir}/bin/python" "${repo_dir}/scripts/run_distillm_baseline.py" \
    --config "${config_path}" \
    --scratch-root "${scratch_root}" \
    --registry "${repo_dir}/configs/baselines.yaml" \
    --manifest "${repo_dir}/third_party/manifest.yaml" \
    --official-root "${scratch_root}/official"
fi

metadata_path="$("${venv_dir}/bin/python" -c 'import sys, yaml; from pathlib import Path; payload=yaml.safe_load(Path(sys.argv[1]).read_text()); print(Path(sys.argv[2]) / payload["output_dir"] / "run_metadata.json")' "${config_path}" "${scratch_root}")"
if [[ -n "${BASELINE_ID:-}" ]] && [[ "${training_backend}" == "trl" || "${training_backend}" == "opsd" ]]; then
  metadata_path="${scratch_root}/checkpoints/${BASELINE_ID}-${run_suffix}/run_metadata.json"
fi
cat "${metadata_path}"
echo "end_time=$(date --iso-8601=seconds)"
