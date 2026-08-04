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
- Repository code for the corrected matrix: commit `7532b52` (the launcher and
  LoRA-routing fixes are in `176ef1a`; the current matrix dispatch/report
  bookkeeping is in the newer commit).

## 5. Live official-run status

- Job 18789: D2 NoveltyBench smoke completed and passed. It produced a valid 2-prompt ×
  2-generation summary under
  `/scratch/node02/aryama.murthy/novelty-distill/evaluations/official/official-smoke-D2-seed17/noveltybench/summary.json`.
- Job 18788: A0 smoke was rejected because a concurrent worker mutated the shared Python
  environment during import; no benchmark result was accepted.
- Jobs 18792–18878 were stopped before acceptance after a routing audit found that the pinned
  SGLang OpenAI API requires `base-model:adapter-name` to activate a LoRA. Those pre-fix jobs
  could silently serve the base model; their partial artifacts are retained under
  `evaluations/official/invalid-pre-lora-selection-20260804/` and are excluded from analysis.
- The corrected launcher uses `novelty-base:novelty-model` for every LoRA request and leaves A0
  on `novelty-model`. A direct live request on the corrected B1 server produced different text
  for `novelty-base:novelty-model` versus `novelty-base`, confirming adapter activation.
- Corrected A0 jobs: 18883–18887. The first LoRA restart (18888–18917) was stopped after
  HypoSpace exposed the branch-scoped export bug; it is retained under the second invalid
  archive. The current corrected LoRA jobs are B1/B2b on node02 (18919–18928) and C1-best1,
  C2-best1, D1, and D2 on node03 (18975–18994). No result is accepted until every LoRA HypoSpace
  artifact records `OpenRouter(novelty-base:novelty-model)` and all four summaries per model
  validate and combine successfully.

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

## 8. Numerical appendix: production and target construction

The teacher generation pass completed with exactly 1,000 prompt shards and 8,000 completions.
Every prompt had eight distinct completions; 7,985 completions ended normally and 15 reached the
512-token cap. The frozen teacher quality aggregate was 0.827656 (population SD 0.104174), with
feasibility mean 3.733625 and soundness mean 3.71425 on the five-point rubric. Instruction
compliance was near a ceiling (99.25% rating-5), so it is not a useful discriminator by itself.

The target-release clustering audit selected complete-linkage threshold 0.94 after the earlier
0.82 threshold produced only about one mode per prompt. The released target has 3.798 instructed
modes per prompt on average (median 4), and 54.2% of prompts have at least four measured modes.
The diverse-4 view contains 3.146 measured modes per prompt on average. Best-1 quality averages
0.9166 versus 0.866088 for diverse-4, which is the intended quality-versus-breadth tradeoff.

## 9. Numerical appendix: training systems costs

These are measured TOMATO-1k production values from the run ledger. “Available rows” counts the
static prompt-target rows made available to the trainer; “exposures” is the compute-matched number
of optimizer examples consumed.

| Baseline | Objective/view | Available rows | Exposures | Runtime (s) | Peak GPU MiB |
|---|---|---:|---:|---:|---:|
| B1 | CE / human | 1,000 | 1,000 | 305.3 | 15,286 |
| B2a | CE / random-1 | 1,000 | 1,000 | 245.9 | 10,902 |
| B2b | CE / best-1 | 1,000 | 1,000 | 245.1 | 10,982 |
| B2c | CE / mode-1 | 1,000 | 1,000 | 251.0 | 10,924 |
| B3 | CE / diverse-4 | 4,000 | 1,000 | 240.8 | 10,924 |
| B4 | GEM / diverse-4 | 4,000 | 1,000 | 340.9 | 39,626 |
| C1-human | forward KL / human | 1,000 | 1,000 | 617.7 | 45,662 |
| C1-best1 | forward KL / best-1 | 1,000 | 1,000 | 487.2 | 40,298 |
| C1-diverse4 | forward KL / diverse-4 | 4,000 | 1,000 | 485.8 | 40,498 |
| C2-human | reverse KL / human | 1,000 | 1,000 | 617.0 | 43,020 |
| C2-best1 | reverse KL / best-1 | 1,000 | 1,000 | 489.9 | 40,024 |
| C2-diverse4 | reverse KL / diverse-4 | 4,000 | 1,000 | 487.7 | 40,078 |
| D1 | on-policy forward KL | 1,000 prompt exposures | 1,000 | 13,524.8 | ~39,500 |
| D2 | on-policy reverse KL | 1,000 prompt exposures | 1,000 | 13,638.4 | ~39,500 |

All six production compact baselines are Qwen3-4B rank-16 LoRA adapters with 132,187,888-byte
validated final artifacts. D1/D2 are not static-target datasets: they generate student
trajectories during training, then evaluate the teacher distribution on those trajectories.

## 10. Numerical appendix: completed TOMATO held-out results

The frozen compact evaluation used 1,658 held-out prompts, K=4 generations per prompt, and a
separate Qwen3-32B-FP8 judge. The columns are feasibility (/5), soundness (/5), teacher-mode
recall, and viable semantic yield (/4).

| Method | Feasibility | Soundness | Teacher recall | Viable yield |
|---|---:|---:|---:|---:|
| A0 | 4.218 | 4.603 | 0.120 | 1.900 |
| B1 | 3.636 | 3.603 | 0.025 | 1.799 |
| B2b | 3.398 | 3.176 | 0.031 | 0.565 |
| C1-best1 | 4.239 | 4.657 | 0.134 | 2.248 |
| C2-best1 | 4.240 | 4.656 | 0.135 | 1.795 |
| D1 | 4.234 | 4.658 | 0.117 | 2.223 |
| D2 | 4.231 | 4.638 | 0.125 | 1.737 |

The principal predeclared findings are: B1 is below A0 on all four outcomes; B2b sharply reduces
viable breadth; C1-best1 recovers quality and breadth; C2-best1 is similar in quality but more
mode-seeking; and D1 does not improve over C1-best1. The separately predeclared D2 extension
found no meaningful feasibility or teacher-recall difference from D1/C1/C2, but lower soundness
and viable yield. D2 viable yield was 0.486 below D1 and 0.511 below C1-best1.

These are baseline-study findings under a frozen operational judge. They are not human validation
of scientific novelty and are not a claim that the judge is a novelty oracle.

## 11. External benchmark accounting

The external evaluation is deliberately separate from the training prompts and TOMATO judge:

- NoveltyBench: 100 curated prompts × 10 generations = 1,000 generated answers per model.
- HypoSpace causal: 61 tasks × 10 queries = 610 answers per model.
- HypoSpace 3D: 9 tasks × 10 queries = 90 answers per model.
- HypoSpace Boolean: 35 tasks × 10 queries = 350 answers per model.
- Total: 2,050 generated answers per model, 14,350 across seven models.

Every suite uses seed 17, temperature 0.7, top-p 0.9, max tokens 512, the same pinned source
revisions, and one GPU. NoveltyBench reports `distinct_k` and `utility_k`. HypoSpace reports
parse completion, validity, uniqueness/novelty, and recovery. The pinned Boolean schema does not
emit a parser-stage statistic; because it reports zero provider errors and complete per-sample
records, the canonical adapter records parser completion as 1.0 and documents that fallback.

## 12. Current official job manifest

| Model | NoveltyBench | Causal | 3D | Boolean | Combine |
|---|---:|---:|---:|---:|---:|
| A0 | 18883 | 18884 | 18885 | 18886 | 18887 |
| B1 (node02) | 18919 | 18920 | 18921 | 18922 | 18923 |
| B2b (node02) | 18924 | 18925 | 18926 | 18927 | 18928 |
| C1-best1 (node03) | 18975 | 18976 | 18977 | 18978 | 18979 |
| C2-best1 (node03) | 18980 | 18981 | 18982 | 18983 | 18984 |
| D1 (node03) | 18985 | 18986 | 18987 | 18988 | 18989 |
| D2 (node03) | 18990 | 18991 | 18992 | 18993 | 18994 |

Jobs 18827–18846 were duplicate pending submissions caused by an asynchronous launcher-output
race and were cancelled before execution. Jobs 18792, 18848, 18850, and 18853 were stopped after
the first LoRA-routing audit; their artifacts are excluded. The first corrected LoRA restart
18888–18917 was also stopped after the HypoSpace export-scope audit; its B1 causal summary is
retained only in `invalid-pre-lora-selection-20260804-r2/`. The current matrix above is the only
accepted run set once validation completes. Node03 dispatch 18953–18972 failed before execution
because its repository checkout had not yet been staged; those jobs produced no accepted artifacts.
