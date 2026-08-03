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

For a promoted checkpoint, the complete official suite uses four independent GPU jobs: all 100
curated NoveltyBench prompts with K=10 and all 61 causal, 9 3D, and 35 Boolean HypoSpace cases with
10 queries each. Results are content-bound, resumable, namespaced by `EVAL_ID`, and combined only
after all four official artifacts validate:

```bash
EVAL_ID=<method-run> MODEL_PATH=Qwen/Qwen3-4B LORA_PATH=<adapter-final> \
  scripts/submit_official_model_evaluation.sh
```

Full checkpoints use `MODEL_PATH=<checkpoint>` without `LORA_PATH`. The launcher registers adapters
under the fixed `novelty-model` name required by the pinned official clients.

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

The complete research-scale teacher, target, and training graph is submitted with bounded resume
passes by:

```bash
scripts/submit_tomato1k_pipeline.sh
```

After the 1k gate selects the promoted methods, prepare either larger teacher bank with four
disjoint GPU workers per model-serving stage and fail-closed global validation between stages:

```bash
TRAIN_SIZE=5000 scripts/submit_teacher_scale.sh
TRAIN_SIZE=20000 scripts/submit_teacher_scale.sh
```

Each scale chain is resumable and produces one canonical generation bank, score bank, merged
cluster artifact, teacher-target artifact, and final target-analysis gate. The 20k command is not a
replacement for the 5k method-selection stage. The 5k bank first imports the exact compatible 1k
shards, and the 20k bank imports the 5k shards, so nested prompts are never regenerated.

Once the preregistered analysis has selected matrix indexes, launch only those promoted methods;
index 5 (GEM) and index 12 (DistiLLM) are automatically split into their required allocations:

```bash
TRAIN_SIZE=5000 TARGET_GATE_JOB_ID=<5k-gate> PROMOTED_INDICES=<indexes> \
  scripts/submit_promoted_training.sh
TRAIN_SIZE=20000 TARGET_GATE_JOB_ID=<20k-gate> PROMOTED_INDICES=<indexes> \
  scripts/submit_promoted_training.sh
```

The 5k default is seed 17. The 20k default is the frozen central seed set 17, 29, and 43; neither
launcher chooses winners before the preceding analysis.

The 1.7B/8B replication is a separate model profile with its own Qwen3-8B teacher samples and
targets. Start it only after the main study works, beginning with its independent 1k gate:

```bash
TEACHER_PROFILE=replication TRAIN_SIZE=1000 START_AFTER_JOB_ID=<main-analysis> \
  scripts/submit_teacher_scale.sh
MODEL_PROFILE=replication TRAIN_SIZE=1000 TARGET_GATE_JOB_ID=<replication-target-gate> \
  PROMOTED_INDICES=<central-method-indexes> scripts/submit_promoted_training.sh
```

The same launchers extend that profile to 5k and 20k while reusing only nested Qwen3-8B shards.
Replication outputs use `qwen1p7b` names and are never pooled with the 4B/14B results.

The post-freeze DRKL treatments are isolated from the primary matrix and can be staged behind a
chosen gate with:

```bash
START_AFTER_JOB_ID=<primary-gate> scripts/submit_exploratory_drkl.sh
```

Their attribution, matched controls, and secondary-only interpretation are frozen in
`reports/EXPLORATORY_TREATMENTS.md`.

All jobs keep environments, Hugging Face caches, data, generations, and checkpoints under
`/scratch/$USER/novelty-distill`, including Slurm logs. Turing scratch is node-local, so the
submission helper stages the small batch script under `~/.cache/novelty-distill-submit` and calls
`sbatch` directly; it does not create an interactive control allocation. Each compute job then
synchronizes the pinned branch under a shared repository lock.

Final primary and exploratory analyses also backfill the pinned W&B SDK from immutable JSON
provenance. The default is credential-free `WANDB_MODE=offline`; configs, metric tables, and small
metadata artifacts stay under `/scratch/$USER/novelty-distill/wandb`, while model weights remain in
their content-bound checkpoint paths. Set `WANDB_MODE=online`, `WANDB_PROJECT`, and optionally
`WANDB_ENTITY` only when live upload is intended. Export manifests make identical reruns no-ops.

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
The research-scale Slurm graph is frozen in
[`reports/RESEARCH_RUN_LEDGER.md`](reports/RESEARCH_RUN_LEDGER.md), and the claim boundary informed
by current primary literature is in
[`reports/RESEARCH_VALIDITY_AUDIT.md`](reports/RESEARCH_VALIDITY_AUDIT.md).
