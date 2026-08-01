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

cleanup() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}"
    wait "${monitor_pid}" || true
  fi
}
trap cleanup EXIT

mkdir -p \
  "${scratch_root}/logs" \
  "${scratch_root}/cache/torch-extensions" \
  "${scratch_root}/cache/triton" \
  "${scratch_root}/cache/cuda" \
  "${scratch_root}/cache/xdg" \
  "${scratch_root}/tmp"
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
export NOVELTY_GIT_COMMIT
NOVELTY_GIT_COMMIT="$(git -C "${repo_dir}" rev-parse HEAD)"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  uv venv --python 3.10 "${venv_dir}"
fi
"${venv_dir}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
uv pip sync --python "${venv_dir}/bin/python" "${environment_lock}"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

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
    opsd_args+=(--baseline-id "${BASELINE_ID}")
  fi
  "${venv_dir}/bin/python" "${repo_dir}/scripts/run_opsd_baseline.py" "${opsd_args[@]}"
elif [[ "${training_backend}" == "trl" ]]; then
  trl_args=(
    --config "${config_path}"
    --scratch-root "${scratch_root}"
    --registry "${repo_dir}/configs/baselines.yaml"
  )
  if [[ -n "${BASELINE_ID:-}" ]]; then
    trl_args+=(--baseline-id "${BASELINE_ID}")
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
  metadata_path="${scratch_root}/checkpoints/${BASELINE_ID}-smoke/run_metadata.json"
fi
cat "${metadata_path}"
echo "end_time=$(date --iso-8601=seconds)"
