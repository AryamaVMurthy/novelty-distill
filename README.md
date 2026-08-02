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
scripts/turing_submit.sh slurm/prepare_data.sbatch
scripts/turing_submit.sh slurm/sglang_smoke.sbatch
scripts/turing_submit.sh slurm/bootstrap_official.sbatch
scripts/turing_submit.sh slurm/train_smoke.sbatch
scripts/turing_submit.sh slurm/train_smoke.sbatch --export=ALL,TRAINING_BACKEND=trl,RUN_CONFIG=configs/training/gkd_smoke.yaml
scripts/turing_submit.sh slurm/train_smoke.sbatch --export=ALL,TRAINING_BACKEND=opsd
scripts/turing_submit.sh slurm/train_smoke.sbatch --array=0-15 --export=ALL,BASELINE_MATRIX=1
scripts/turing_submit.sh slurm/prepare_hypospace.sbatch
scripts/turing_submit.sh slurm/evaluate_official.sbatch --export=ALL,EVAL_SUITE=noveltybench
scripts/turing_submit.sh slurm/evaluate_official.sbatch --export=ALL,EVAL_SUITE=hypospace,HYPOSPACE_DOMAIN=causal
```

Teacher calibration uses one SGLang server for the frozen seven-condition matrix, followed by the
pinned judge and embedding model:

```bash
scripts/turing_submit.sh slurm/sglang_smoke.sbatch --export=ALL,CALIBRATION_CONFIG=configs/generation/teacher_calibration.yaml,CALIBRATION_RUN_ID=qwen3-14b-v1
scripts/turing_submit.sh slurm/score_teacher.sbatch --export=ALL,CALIBRATION_RUN_ID=qwen3-14b-v1
scripts/turing_submit.sh slurm/cluster_teacher.sbatch --export=ALL,CALIBRATION_RUN_ID=qwen3-14b-v1
```

After calibration promotes `configs/generation/teacher.yaml`, the resumable 1,000-prompt teacher
run uses the same SGLang job with a stable output ID:

```bash
scripts/turing_submit.sh slurm/sglang_smoke.sbatch --export=ALL,GENERATION_CONFIG=configs/generation/teacher.yaml,OUTPUT_NAME=teacher,OUTPUT_RUN_ID=teacher-1k-v1,INPUT_LIMIT=1000,GENERATION_CONCURRENCY=8
```

All jobs keep environments, Hugging Face caches, data, generations, and checkpoints under
`/scratch/$USER/novelty-distill`, including Slurm logs. Turing scratch is node-local, so the
submission helper briefly allocates node01 before calling `sbatch`.

GEM and DistiLLM additionally require the versioned teacher-target artifact produced from the
permanent eight-sample teacher generation set; their launchers intentionally fail rather than
substitute human targets.

Teacher samples are scored by the official `Qwen/Qwen3-32B-FP8` checkpoint served with SGLang,
then clustered with the official `Qwen/Qwen3-Embedding-4B` checkpoint. The FP8 judge is Qwen's
official memory-feasible form of the planned 32B judge for Turing's 48 GB GPU.

NoveltyBench runs directly from its pinned official isolated package and uses Inspect's native
SGLang provider. HypoSpace data and metrics run from the pinned official repository; the only
adapter points its existing OpenRouter-compatible client at the local SGLang URL.

The verified Turing smoke results and their limitations are recorded in
[`reports/TURING_SMOKE_FINDINGS.md`](reports/TURING_SMOKE_FINDINGS.md).
