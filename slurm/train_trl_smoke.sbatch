#!/bin/bash
#SBATCH --job-name=nd-trl-smoke
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
venv_dir="${scratch_root}/venvs/training"
run_config="${RUN_CONFIG:-configs/training/sft_smoke.yaml}"
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
export TMPDIR="${scratch_root}/tmp"
export TOKENIZERS_PARALLELISM=false

cleanup() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}"
    wait "${monitor_pid}" || true
  fi
}
trap cleanup EXIT

mkdir -p "${scratch_root}/logs" "${scratch_root}/tmp"
git -C "${repo_dir}" diff --quiet
git -C "${repo_dir}" fetch origin "${repo_ref}"
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
export NOVELTY_GIT_COMMIT
NOVELTY_GIT_COMMIT="$(git -C "${repo_dir}" rev-parse HEAD)"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  uv venv --python 3.12 "${venv_dir}"
fi
uv pip sync --python "${venv_dir}/bin/python" "${repo_dir}/environments/training.lock"
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_dir}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "git_commit=${NOVELTY_GIT_COMMIT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "start_time=$(date --iso-8601=seconds)"
nvidia-smi
"${venv_dir}/bin/python" -c 'import torch, transformers, trl, peft; print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0)})'

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 5 >"${monitor_file}" &
monitor_pid="$!"

"${venv_dir}/bin/python" "${repo_dir}/scripts/run_trl_baseline.py" \
  --config "${config_path}" \
  --scratch-root "${scratch_root}" \
  --registry "${repo_dir}/configs/baselines.yaml"

metadata_path="$("${venv_dir}/bin/python" -c 'import sys, yaml; from pathlib import Path; payload=yaml.safe_load(Path(sys.argv[1]).read_text()); print(Path(sys.argv[2]) / payload["output_dir"] / "run_metadata.json")' "${config_path}" "${scratch_root}")"
cat "${metadata_path}"
echo "end_time=$(date --iso-8601=seconds)"
