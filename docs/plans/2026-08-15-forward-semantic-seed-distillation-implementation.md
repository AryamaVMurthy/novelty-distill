# Forward Semantic Seed Distillation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and evaluate a compact Turing search over student-coverage target selection, input seed conditioning, and teacherless lookahead auxiliary training on top of the existing off-policy forward-KL baseline.

**Architecture:** Keep the official TRL GKD forward-KL path as the primary loss. Add deterministic conditioning and target-selection modules outside the model, plus an optional audited teacherless auxiliary pass inside a narrow GKD subclass. Use successive-halving Slurm jobs so only candidates that preserve judged validity reach full TOMATO-1k confirmation.

**Tech Stack:** Python 3.10, Pydantic, NumPy, Hugging Face Transformers, TRL GKD, PEFT LoRA, SGLang, pytest, Slurm/Turing.

---

### Task 1: Deterministic input seed representation

**Files:**
- Create: `src/novelty_distill/data/semantic_seeds.py`
- Create: `tests/test_semantic_seeds.py`

**Step 1: Write failing tests**

Cover these exact contracts:

```python
def test_gaussian_seed_is_reproducible_and_sample_specific(): ...
def test_gaussian_seed_quantization_is_bounded(): ...
def test_seed_prefix_contains_no_task_or_target_text(): ...
def test_same_seed_renders_identically_for_training_and_generation(): ...
def test_invalid_dimensions_scale_and_bins_fail_closed(): ...
```

Use a frozen expected prefix such as:

```text
Exploration seed: G2:N0:P1:N2
Use this arbitrary seed only to choose one coherent approach; do not mention it.
```

**Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_semantic_seeds.py -q`

Expected: import failure for `novelty_distill.data.semantic_seeds`.

**Step 3: Implement the minimal module**

Implement:

```python
class GaussianSeedSpec(BaseModel):
    dimensions: int = Field(gt=0, le=32)
    bins: int = Field(ge=2, le=9)
    scale: float = Field(gt=0, le=4)
    salt: str = Field(min_length=1)

def gaussian_seed_values(*, prompt_id: str, sample_index: int,
                         generation_seed: int,
                         spec: GaussianSeedSpec) -> tuple[int, ...]: ...

def render_seed_prefix(values: Sequence[int]) -> str: ...

def condition_prompt(prompt: str, *, values: Sequence[int]) -> str: ...
```

Seed NumPy's `PCG64` from a SHA-256 digest of the complete tuple. Quantize
standard-normal draws by fixed standard-normal quantile cut points; never fit
bins on evaluation data.

**Step 4: Verify tests**

Run: `uv run pytest tests/test_semantic_seeds.py -q`

Expected: all pass.

**Step 5: Commit**

```bash
git add src/novelty_distill/data/semantic_seeds.py tests/test_semantic_seeds.py
git commit -m "feat: add deterministic Gaussian input seeds"
```

### Task 2: Condition both training and generation prompts

**Files:**
- Modify: `src/novelty_distill/training/trl.py`
- Modify: `src/novelty_distill/generation/sglang.py`
- Modify: `tests/test_trl_training.py`
- Modify: `tests/test_sglang.py`

**Step 1: Add failing config and parity tests**

Add optional fields to the test expectations:

```python
input_seed: GaussianSeedSpec | None = None
```

Test that:

- training row `sample_index=i` and generation request `sample_index=i` render
  the identical prefix;
- unconditioned configs retain byte-identical prompts and fingerprints;
- the generation fingerprint changes when seed settings change;
- conditioned generation remains one request per sample and continues to use
  `spec.seed + sample_index` for token sampling;
- metadata includes the fully rendered seed specification.

**Step 2: Verify focused failure**

Run:

```bash
uv run pytest tests/test_trl_training.py tests/test_sglang.py -q
```

Expected: missing `input_seed` support.

**Step 3: Implement prompt conditioning**

- Add `input_seed` to `TRLRunSpec` and `GenerationSpec`.
- Pass a target/sample index through `build_trl_rows`.
- Change `render_generation_prompt` to accept `sample_index`.
- Apply the seed before response requirements so the scientific-task contract
  remains the final instruction.
- Record seed controls in `run_metadata.json` and generation manifests.

**Step 4: Verify compatibility**

Run:

```bash
uv run pytest tests/test_trl_training.py tests/test_sglang.py \
  tests/test_training_provenance.py tests/test_provenance.py -q
```

Expected: all pass, including old unconditioned fixtures.

**Step 5: Commit**

```bash
git add src/novelty_distill/training/trl.py \
  src/novelty_distill/generation/sglang.py \
  tests/test_trl_training.py tests/test_sglang.py
git commit -m "feat: condition distillation and generation on input seeds"
```

### Task 3: Build student-coverage-residual teacher targets

**Files:**
- Create: `src/novelty_distill/data/coverage_residual.py`
- Create: `scripts/derive_coverage_residual_targets.py`
- Create: `tests/test_coverage_residual.py`

**Step 1: Write failing selector tests**

Construct tiny normalized embeddings and prove:

- an invalid outlier is never selected;
- between equally valid candidates, the one farther from student probes wins;
- increasing `gamma` strengthens undercoverage weighting;
- `gamma=0` reduces to quality-only selection;
- cosine/vMF density is finite for duplicate points;
- ties resolve by canonical sample index;
- the output reconstructs the existing teacher-target schema and contains
  source-artifact hashes and selector parameters.

**Step 2: Run and verify failure**

Run: `uv run pytest tests/test_coverage_residual.py -q`

Expected: missing module.

**Step 3: Implement the selector**

Implement:

```python
def vmf_similarity_density(teacher, student, *, kappa: float) -> np.ndarray: ...

def residual_weights(quality, density, *, minimum_quality: float,
                     gamma: float, epsilon: float) -> np.ndarray: ...

def select_residual_target(candidates, student_embeddings, parameters): ...
```

Require normalized finite arrays, at least one valid teacher candidate, and a
non-empty student probe set. Do not silently fall back to `best1`.

**Step 4: Implement the CLI artifact builder**

The CLI accepts the frozen teacher sample/score/embedding bank and untouched
student probe embeddings, verifies exact prompt/sample coverage, and writes a
versioned target artifact plus audit JSON. It supports only the declared grid:
`kappa in {8, 16, 32}`, `gamma in {0.5, 1.0}`, and a fixed validity threshold.

**Step 5: Verify**

Run:

```bash
uv run pytest tests/test_coverage_residual.py tests/test_teacher_views.py \
  tests/test_target_integrity.py -q
```

Expected: all pass.

**Step 6: Commit**

```bash
git add src/novelty_distill/data/coverage_residual.py \
  scripts/derive_coverage_residual_targets.py tests/test_coverage_residual.py
git commit -m "feat: derive student-coverage residual targets"
```

### Task 4: Add the teacherless lookahead auxiliary tensors and loss

**Files:**
- Create: `src/novelty_distill/training/lookahead.py`
- Create: `tests/test_lookahead.py`

**Step 1: Write failing unit tests**

Test that:

- auxiliary inputs retain every prompt token;
- every active completion input is the declared neutral token;
- labels exactly equal the ordinary completion labels;
- prompt and truncated-tail labels remain `-100`;
- CE is finite and zero-weight returns an exact scalar zero;
- a malformed/no-active-label batch fails closed.

**Step 2: Verify failure**

Run: `uv run pytest tests/test_lookahead.py -q`

Expected: missing module.

**Step 3: Implement pure tensor helpers**

Implement framework-light construction first and import Torch only within loss
functions:

```python
def build_teacherless_inputs(input_ids, labels, *, neutral_token_id): ...
def teacherless_cross_entropy(logits, labels): ...
```

The helper must never substitute target tokens into the auxiliary inputs.

**Step 4: Verify**

Run: `uv run pytest tests/test_lookahead.py -q`

Expected: all pass or Torch-specific tests skip only when Torch is unavailable.

**Step 5: Commit**

```bash
git add src/novelty_distill/training/lookahead.py tests/test_lookahead.py
git commit -m "feat: add teacherless lookahead objective primitives"
```

### Task 5: Integrate the optional lookahead pass into TRL GKD

**Files:**
- Modify: `src/novelty_distill/training/trl.py`
- Modify: `tests/test_trl_training.py`
- Modify: `tests/test_training_provenance.py`

**Step 1: Add failing integration tests**

Add `teacherless_weight: float = 0` and `teacherless_neutral_token: str | None`
to `TRLRunSpec`. Prove that:

- weight zero selects the untouched official `GKDTrainer`;
- positive weight selects a local subclass and requires a single-token neutral
  string;
- the auxiliary forward is performed after the ordinary loss;
- total loss is `ordinary + weight * auxiliary`;
- run metadata records weight, token string/ID, and component losses;
- SFT and on-policy GKD reject the auxiliary mode in the first implementation.

**Step 2: Verify failure**

Run:

```bash
uv run pytest tests/test_trl_training.py tests/test_training_provenance.py -q
```

Expected: run-spec validation/integration failures.

**Step 3: Implement the narrow trainer subclass**

Subclass the pinned TRL `GKDTrainer` only for off-policy teacher trajectories.
Call the official loss path first. Run a second student forward with the
teacherless tensors, compute CE, and add the weighted auxiliary. Do not include
the 14B teacher in the auxiliary pass. Log both loss components.

**Step 4: Verify locally**

Run:

```bash
uv run pytest tests/test_trl_training.py tests/test_lookahead.py \
  tests/test_training_provenance.py -q
uv run ruff check src/novelty_distill/training tests/test_lookahead.py
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/novelty_distill/training/trl.py \
  tests/test_trl_training.py tests/test_training_provenance.py
git commit -m "feat: integrate teacherless lookahead with forward KL"
```

### Task 6: Freeze the exploratory candidate registry and short configs

**Files:**
- Create: `configs/semantic_seed_baselines.yaml`
- Create: `configs/training/semantic_seed_short.yaml`
- Create: `configs/generation/semantic_seed_short.yaml`
- Create: `tests/test_semantic_seed_configs.py`

**Step 1: Write failing config tests**

Require exactly these initial candidates:

```text
SS0-C1
SS1-CR
SS2-GSC
SS3-FPS
SS4-TL
```

All use off-policy teacher trajectories and forward KL. Freeze 128 training
examples, 32 steps, seed 17, and matched optimizer exposure. Generation uses a
disjoint 128-prompt slice and `K=4`.

**Step 2: Verify failure**

Run: `uv run pytest tests/test_semantic_seed_configs.py -q`

Expected: missing configs.

**Step 3: Add configs and validation**

Use the existing Qwen3-4B/Qwen3-14B revisions, context length, LoRA rank, and
optimizer controls. Only candidate-specific seed, target artifact, or
teacherless weight fields may differ.

**Step 4: Verify**

Run:

```bash
uv run pytest tests/test_semantic_seed_configs.py tests/test_baselines.py \
  tests/test_tomato1k_configs.py -q
```

Expected: all pass.

**Step 5: Commit**

```bash
git add configs/semantic_seed_baselines.yaml \
  configs/training/semantic_seed_short.yaml \
  configs/generation/semantic_seed_short.yaml \
  tests/test_semantic_seed_configs.py
git commit -m "config: freeze semantic seed short-run matrix"
```

### Task 7: Add short-run slice and selection analysis

**Files:**
- Create: `scripts/prepare_semantic_seed_short_data.py`
- Create: `scripts/analyze_semantic_seed_short.py`
- Create: `tests/test_semantic_seed_short_analysis.py`

**Step 1: Write failing tests**

Prove deterministic disjoint train/eval prompt selection, seed-stratified
metrics, same-seed repeat handling, validity-first ranking, and rejection when
quality drops exceed 0.10.

**Step 2: Implement data slicing**

Select prompt IDs by SHA-256 rank, not file position, and write manifests that
bind the source hash and exact IDs. Never inspect held-out scores during slice
construction.

**Step 3: Implement selection analysis**

Report:

- feasibility, soundness, normalized quality;
- semantic clusters and quality-qualified yield;
- between-seed distance minus repeated-same-seed distance;
- completion length and stop rate;
- eligibility status and deterministic rank among eligible candidates.

**Step 4: Verify**

Run:

```bash
uv run pytest tests/test_semantic_seed_short_analysis.py -q
uv run ruff check scripts/prepare_semantic_seed_short_data.py \
  scripts/analyze_semantic_seed_short.py
```

Expected: all pass.

**Step 5: Commit**

```bash
git add scripts/prepare_semantic_seed_short_data.py \
  scripts/analyze_semantic_seed_short.py \
  tests/test_semantic_seed_short_analysis.py
git commit -m "feat: analyze semantic seed successive-halving runs"
```

### Task 8: Add fail-closed Turing launchers

**Files:**
- Create: `slurm/prepare_semantic_seed_short.sbatch`
- Create: `slurm/train_semantic_seed_short.sbatch`
- Create: `slurm/evaluate_semantic_seed_short.sbatch`
- Create: `slurm/analyze_semantic_seed_short.sbatch`
- Create: `scripts/submit_semantic_seed_short.sh`
- Create: `tests/test_semantic_seed_slurm.py`

**Step 1: Write static contract tests**

Require the canonical account/partition/QoS, node-local scratch, repository and
environment locks, GPU telemetry, bounded arrays, explicit dependencies,
content-bound run IDs, and no home-directory model/cache output.

**Step 2: Verify failure**

Run: `uv run pytest tests/test_semantic_seed_slurm.py -q`

Expected: missing launchers.

**Step 3: Implement the graph**

The controller submits:

```text
prepare -> train[eligible candidates] -> generate -> judge -> embed/evaluate -> analyze
```

Reuse existing generation, scoring, and embedding commands. The analysis job
must require complete manifests for every candidate before selecting one.

**Step 4: Verify all local checks**

Run:

```bash
uv run pytest -q
uv run ruff check .
bash -n scripts/submit_semantic_seed_short.sh slurm/*.sbatch
```

Expected: full suite passes.

**Step 5: Commit and push**

```bash
git add slurm/prepare_semantic_seed_short.sbatch \
  slurm/train_semantic_seed_short.sbatch \
  slurm/evaluate_semantic_seed_short.sbatch \
  slurm/analyze_semantic_seed_short.sbatch \
  scripts/submit_semantic_seed_short.sh tests/test_semantic_seed_slurm.py
git commit -m "feat: launch semantic seed short-run search"
git push origin codex/implementation
```

### Task 9: Repair and validate Turing staging

**Files:**
- Modify if required: `slurm/repair_staged_repository.sbatch`
- Test: `tests/test_repository_sync_contracts.py`

**Step 1: Resolve exact repair targets read-only**

Verify Turing identity, queue, node scratch repositories, and preserved target
paths. Do not alter unrelated home projects or research outputs.

**Step 2: Repair only staged repository worktrees**

Submit one CPU-only repair per node whose staging checkout is dirty. Preserve
the old checkout under a unique content-addressed directory, clone the pushed
branch into the canonical path, and assert the exact expected commit and clean
status.

**Step 3: Validate a one-step GPU smoke**

Submit `SS0-C1` and each novel code path with one example/one step. Require
CUDA use, finite losses, loadable adapters, recorded seed/auxiliary metadata,
and zero prompt truncation before continuing.

**Step 4: Record evidence**

Write job IDs, accounting output, log paths, config hashes, and checkpoint
hashes to `reports/semantic-seed-short/RUN_AUDIT.md`.

### Task 10: Run Gate 1 and Gate 2, then iterate

**Files:**
- Create: `reports/semantic-seed-short/findings.md`
- Create: `reports/semantic-seed-short/selection.json`

**Step 1: Submit Gate 1 decoding controls**

Compare ordinary temperature sampling, cold Gaussian-seeded sampling, and
future-code seeded sampling on 64 prompts with `K=8`, including same-seed
repeats.

**Step 2: Analyze Gate 1**

Reject prefix formats that reduce feasibility or soundness by more than 0.10 or
do not create a measurable seed-dependent semantic effect.

**Step 3: Submit Gate 2 short trainings**

Run the surviving candidates and matched `SS0-C1` on the frozen 128-prompt
training/evaluation slices.

**Step 4: Analyze and perform one bounded iteration**

Change only one causal knob based on failure evidence: seed length/quantization,
`kappa/gamma`, or teacherless weight. Record the original and revised method;
never overwrite artifacts. Stop after one revision unless the result crosses a
predeclared gate.

**Step 5: Commit compact evidence**

```bash
git add reports/semantic-seed-short
git commit -m "results: record semantic seed short-run search"
```

### Task 11: Confirm the selected method

**Files:**
- Create: `configs/training/semantic_seed_tomato1k.yaml`
- Create: `scripts/submit_semantic_seed_confirmation.sh`
- Create: `reports/semantic-seed-confirmation/findings.md`

**Step 1: Freeze the winner before confirmation**

Copy the exact selected parameters and artifact hashes into the confirmation
config. The only allowed changes are `max_examples=1000`, `max_steps=125`, and
training seeds 17/29/43.

**Step 2: Run matched three-seed training**

Train selected method and rerun `C1-best1` only if an identical reusable
checkpoint is not already provenance-compatible.

**Step 3: Run corrected TOMATO evaluation**

Use the existing frozen judge, embeddings, thresholds, prompt-level paired
analysis, and feasibility-inclusive qualified-yield gate.

**Step 4: Run transfer suites only after TOMATO success**

Run corrected NoveltyBench and HypoSpace on the selected method. Treat these as
transfer results, not scientific-novelty validation.

**Step 5: Completion audit**

Verify every requested method, job, artifact, metric, hash, seed, and gate from
the design. If no method passes, report the negative result; do not declare a
winner by relative rank alone.
