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
```

Both jobs keep environments, Hugging Face caches, data, and generations under
`/scratch/$USER/novelty-distill`. Only small Slurm logs go to home.
