# Novelty-distillation experiment handoff

Last updated: 2026-08-04 (Asia/Kolkata)

This is the index for the complete experiment history. The official NoveltyBench/HypoSpace
matrix is still in progress; the older TOMATO study is complete and should not be confused with
the external benchmark run.

## 1. Data production

- Training prompts: `data/tomato-open-train-1000.jsonl`, 1,000 prompts.
- Teacher: pinned Qwen3-14B, eight generations per prompt = 8,000 candidate responses.
- Frozen teacher target artifact: `data/teacher-targets-tomato1k-v1.json`.
- Target views: human, random-1, best-1, mode-1, and diverse-4.
- Teacher generation integrity: 1,000 prompt shards, 8,000 completions, 7,985 stop finishes and
  15 length finishes (0.1875% length-stop rate).
- Full production provenance and hashes: `reports/RESEARCH_RUN_LEDGER.md` and
  `reports/TEACHER_SCORE_FINDINGS.md` / `reports/TEACHER_TARGET_FINDINGS.md`.

## 2. Trained compact baselines

All use Qwen3-4B LoRA rank 16, seed 17, and 1,000 TOMATO prompts:

| ID | Training objective | Static training rows / prompt exposures |
|---|---|---:|
| B1 | human-target SFT | 1,000 |
| B2b | hard sequence KD, best teacher target | 1,000 |
| C1-best1 | off-policy forward KL, best teacher target | 1,000 |
| C2-best1 | off-policy reverse KL, best teacher target | 1,000 |
| D1 | on-policy forward KL, live student trajectories | 1,000 prompt exposures |
| D2 | on-policy reverse KL, live student trajectories | 1,000 prompt exposures |

The untouched A0 model is an evaluation-only control. Earlier exploratory variants are documented
in the ledger; diverse-4 variants expose 4,000 rows (four targets per prompt) but are compute-
matched to 1,000 optimizer exposures.

Checkpoint directories on Turing node02:

`/scratch/node02/aryama.murthy/novelty-distill/checkpoints/{B1,B2b,C1-best1,C2-best1,D1,D2}-tomato1k-seed17/final`

Each completed adapter is a validated 132,187,888-byte artifact.

## 3. Completed TOMATO evaluation

The compact K=4 study evaluated A0, B1, B2b, C1-best1, C2-best1, and D1 on 1,658 held-out
TOMATO prompts. D2 was then evaluated under the same frozen protocol. The fixed judge was
Qwen3-32B-FP8; it is separate from the Qwen3-14B training teacher.

Main finding: C1-best1 is the practical baseline. B2b narrows viable breadth; C2-best1 is
quality-preserving but more mode-seeking; D1 does not improve over C1-best1; D2 is also high
quality but has the lowest viable semantic yield among the KL controls.

Canonical reports:

- `reports/CURRENT_RESEARCH_STATUS.md`
- `reports/compact-k4-seed17/README.md`
- `reports/compact-k4-seed17/RUN_AUDIT.md`
- `reports/compact-k4-seed17/findings.md`
- `reports/compact-k4-seed17-d2-extension/README.md`
- `reports/compact-k4-seed17-d2-extension/RUN_AUDIT.md`
- `reports/compact-k4-seed17-d2-extension/findings.md`

## 4. External benchmark plan

For each of A0, B1, B2b, C1-best1, C2-best1, D1, and D2:

- official pinned NoveltyBench: 100 curated prompts × 10 generations;
- official pinned HypoSpace: causal 61 tasks, 3D 9 tasks, Boolean 35 tasks × 10 generations;
- identical seed 17, temperature 0.7, top-p 0.9, max tokens 512, scorer revisions, and one GPU
  per suite component;
- strict sample-count, parser/provider-error, model-identity, artifact-hash, and summary checks;
- four component summaries are combined into one model-level result.

Pinned source revisions:

- Inspect Evals: `6a35510e530f236fd1dbcd9df888f01937c8494a`.
- HypoSpace: `c69e9318577b34b5b896996571aefd4ba6053f58`.
- Repository code: commit `454945f8e72e55e3480cdf8dbec74e8ecf2547d`.

## 5. Live official-run status

- Job 18789: D2 NoveltyBench smoke completed and passed. It produced a valid 2-prompt ×
  2-generation summary under
  `/scratch/node02/aryama.murthy/novelty-distill/evaluations/official/official-smoke-D2-seed17/noveltybench/summary.json`.
- Job 18788: A0 smoke was rejected because a concurrent worker mutated the shared Python
  environment during import; no benchmark result was accepted.
- Job 18791: A0 smoke retry is running with per-job isolated inference and scorer environments.
- Full seven-model submission is gated on job 18791 passing. No full official metric is claimed
  until all four suite summaries per model validate and combine successfully.

## 6. Where raw logs and artifacts live

On Turing node02:

`/scratch/node02/aryama.murthy/novelty-distill/logs/`

contains Slurm logs, SGLang server logs, and Inspect traces. Official results are under:

`/scratch/node02/aryama.murthy/novelty-distill/evaluations/official/`

The repository source, configs, tests, and scripts are in this checkout. The official evaluator
is `slurm/evaluate_official.sbatch`; model-level validation/combination is
`slurm/combine_official_results.sbatch`.

## 7. Interpretation limits

The completed TOMATO results are a baseline/distillation study, not a novelty claim. The external
benchmarks are being used to test transfer beyond the original held-out TOMATO analysis. LLM-judge
scores are not treated as a novelty oracle; diversity, utility, parsing/validity, and failure rates
must be inspected together.
