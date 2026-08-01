# novelty-distill

Reproducible baselines for measuring how supervised and knowledge-distillation methods change scientific-idea quality and mode coverage.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the baseline matrix and Turing execution order.

The project uses official upstream implementations pinned in `third_party/manifest.yaml`. SGLang is the inference engine for teacher generation, evaluation sampling, and judge serving; training remains in the official TRL/OPSD/GEM/DistiLLM stacks.

## Local checks

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

## Turing smoke path

```bash
ssh turing 'mkdir -p "$HOME/logs"'
ssh turing 'cd "$HOME" && sbatch' < slurm/prepare_data.sbatch
ssh turing 'cd "$HOME" && sbatch' < slurm/sglang_smoke.sbatch
ssh turing 'cd "$HOME" && sbatch' < slurm/bootstrap_official.sbatch
ssh turing 'cd "$HOME" && sbatch' < slurm/train_smoke.sbatch
ssh turing 'cd "$HOME" && sbatch --export=ALL,TRAINING_BACKEND=trl,RUN_CONFIG=configs/training/gkd_smoke.yaml' < slurm/train_smoke.sbatch
ssh turing 'cd "$HOME" && sbatch --export=ALL,TRAINING_BACKEND=opsd' < slurm/train_smoke.sbatch
```

All jobs keep environments, Hugging Face caches, data, generations, and checkpoints under
`/scratch/$USER/novelty-distill`. Only small Slurm logs go to home.

GEM and DistiLLM additionally require the versioned teacher-target artifact produced from the
permanent eight-sample teacher generation set; their launchers intentionally fail rather than
substitute human targets.

Teacher samples are scored by the official `Qwen/Qwen3-32B-FP8` checkpoint served with SGLang,
then clustered with the official `Qwen/Qwen3-Embedding-4B` checkpoint. The FP8 judge is Qwen's
official memory-feasible form of the planned 32B judge for Turing's 48 GB GPU.
