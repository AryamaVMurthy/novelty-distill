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

A0 is now fully validated and combined (job 18887). Its external control metrics are NoveltyBench
`distinct_k_mean=1.000`, `utility_k_mean=1.841757`, with all three HypoSpace summaries present and
hash-checked. The remaining six model suites are still in progress; no cross-model external
comparison is reported until they each pass the same four-summary gate.

## 13. Exact task and judging contract

The shared TOMATO generation instruction is:

> Return only one hypothesis and its test plan in at most 300 words. Be specific about the
> mechanism, how it differs from existing approaches, the intervention or experiment, measurable
> outcomes, and what result would falsify the hypothesis. Do not add preambles or repeat the
> research background.

The teacher and student see the same scientific task and response contract during generation. A
student is not evaluated on whether it reproduces the teacher's wording. The teacher bank is used
to define a finite reference distribution and training views; the student is sampled independently
under the frozen decoding settings. The human-target B1 row uses TOMATO's human answer, B2b uses the
best judged teacher answer, and the KL rows use token-level teacher probabilities on the selected
training trajectory. D1/D2 generate a student trajectory online and query the teacher on that
trajectory rather than replaying a static response.

The fixed Qwen3-32B-FP8 judge receives only `Task:` followed by the prompt and `Candidate answer:`
followed by one answer. Its system rubric asks it to score five dimensions independently from 1 to
5: relevance, feasibility, soundness, clarity, and instruction compliance. It explicitly says not
to give a 5 merely for fluency or length and reserves 5 for an exceptional answer with a concrete
mechanism, operational test, and no material gap. It returns strict JSON; the normalized quality
proxy is `(mean(score)-1)/4`. Feasibility and soundness are retained separately, so a high mean
cannot hide a failed dimension.

The viability gate is stricter than the quality mean: relevance, soundness, and clarity must each
be at least 4/5. The `viable_semantic_yield` metric then counts the largest set of mutually
distinct viable answers under the selected embedding threshold. This is deliberately named an
operational yield, not a scientific-novelty score.

The embedding model is Qwen3-Embedding-4B with the instruction “Represent the scientific
hypothesis for clustering by its central mechanism, intervention, and experimental test.” Teacher
embeddings are complete-link clustered first at cosine 0.94. Each student embedding is assigned
to a teacher mode only when its minimum similarity to every member of that mode reaches 0.94;
otherwise it is clustered as a separately named student-novel mode. This prevents student bridge
samples from changing the teacher partition.

## 14. Metric dictionary and how to read the results

| Metric | Meaning | Good direction | Important limitation |
|---|---|---|---|
| Feasibility | Judge score for an actionable, resource-credible test | higher | LLM rubric proxy, not expert review |
| Soundness | Judge score for coherent mechanism and non-overclaiming conclusion | higher | Can be biased by fluent presentation |
| Quality | Normalized mean of all five rubric dimensions | higher | Not novelty; instruction compliance is near ceiling |
| Teacher-mode recall | Fraction of sampled teacher modes represented by students | higher | Only covers eight teacher samples, not all valid ideas |
| Teacher-mode precision | Fraction of student modes assigned to teacher modes | higher | Depends on the 0.94 embedding boundary |
| Cluster JSD | Plug-in divergence between teacher/student mode counts | lower | Finite-sample, threshold- and budget-dependent |
| Semantic clusters | Number of distinct embedding clusters in student outputs | descriptive | Raw clusters may be low-quality or paraphrastic |
| Quality-adjusted coverage | Quality-weighted breadth before the viability gate | higher | Still inherits judge and embedding bias |
| Viable semantic yield | Count of distinct outputs passing the quality gate | higher | Operational proxy; not human novelty |
| Length-stop rate | Fraction ending at the 512-token cap | lower | A truncation diagnostic, not a quality metric |

All TOMATO comparisons use paired held-out prompt IDs and treat the prompt, not each completion,
as the statistical unit. The contrast reports use 10,000 prompt-level sign-flip/bootstrap draws,
pointwise 95% intervals, and Holm correction within each declared metric family. A low p-value only
supports a difference in this operational protocol; it does not validate a scientific novelty claim.

## 15. Findings in plain language

The untouched A0 model is already strong on judged quality (4.218 feasibility, 4.603 soundness),
but its four-sample viable yield is only 1.900 per prompt. Human-target SFT (B1) is a negative
control: it falls to 3.636/3.603 and 1.799 yield, with teacher-mode recall dropping from 0.120 to
0.025. This says that simply imitating the human target under this compute budget is not a safe
quality-preserving baseline.

B2b hard sequence KD improves neither quality nor breadth: 3.398 feasibility, 3.176 soundness,
0.031 recall, and only 0.565 viable modes per prompt. Its average completion is 212 tokens versus
329 for A0, so the model is producing shorter, narrower answers. This is the clearest baseline
evidence of response-distribution collapse under best-of-eight sequence imitation.

C1-best1 forward-KL distillation is the strongest practical baseline: 4.239 feasibility, 4.657
soundness, 0.134 recall, and 2.248 viable modes. Relative to B2b, its paired viable-yield gain is
1.683 modes/prompt (95% CI [1.622, 1.745]); relative to A0 it adds about 0.348 viable modes while
also slightly improving judged quality. C1 is therefore the reference for any later proposed
method.

C2-best1 reverse-KL is essentially tied with C1 on feasibility, soundness, and recall, but its
viable yield is lower by 0.453 modes/prompt (95% CI [-0.496, -0.410]). Its semantic-cluster mean
is 2.001 versus C1's 2.491. In this finite-step experiment, reverse KL behaves as the more
mode-seeking/narrow control, but the result is empirical and does not establish a universal theorem
about reverse KL.

D1 on-policy forward KL preserves quality and yield close to C1 (4.234/4.658 and 2.223) but does
not improve it; its teacher recall is slightly lower by 0.0164. D2 on-policy reverse KL also keeps
high quality (4.231/4.638), yet yield falls to 1.737, 0.486 below D1 and 0.511 below C1. Thus the
current ladder supplies both KL directions and both trajectory regimes without claiming a winner
beyond this baseline comparison.

## 16. What has and has not been evaluated

Completed: teacher-bank integrity, target-view construction, six compact LoRA trainings, all
paired K=4 TOMATO held-out generations, Qwen3-32B scoring, embedding clustering, threshold curves,
paired statistical contrasts, and the D2 extension. Also completed are smoke tests for the official
benchmark clients and direct LoRA-routing audits.

In progress: transfer of A0/B1/B2b/C1-best1/C2-best1/D1/D2 to the pinned NoveltyBench and three
HypoSpace domains. These are not scored by the TOMATO Qwen judge. NoveltyBench uses its official
utility/distinctness scorer; HypoSpace uses official domain-specific parse, recovery, uniqueness,
and validity checks. The external results are a generalization check, not a replacement for the
paired TOMATO analysis. They will be appended only after all four component summaries per model
pass strict count, error, identity, and hash validation.

Not completed and intentionally not claimed: expert human judgments of literature-grounded novelty,
retrieval-based novelty against a citation corpus, or a novelty benchmark score derived from the
TOMATO judge. The project is a basic baseline/distillation study, not a novelty claim.

## 17. Reproducibility map

- Dataset/configuration contracts: `configs/data/`, `configs/generation/`, and
  `configs/training/`.
- Baseline registry and objective definitions: `configs/baselines.yaml`.
- Teacher and target provenance: `reports/RESEARCH_RUN_LEDGER.md`,
  `reports/TEACHER_SCORE_FINDINGS.md`, and `reports/TEACHER_TARGET_FINDINGS.md`.
- Completed TOMATO analysis: `reports/compact-k4-seed17/` and
  `reports/compact-k4-seed17-d2-extension/`.
- Strict official launcher: `slurm/evaluate_official.sbatch`.
- Strict four-way combiner: `slurm/combine_official_results.sbatch`.
- Local verification: `uv run --with pytest pytest -q` (282 passed, 2 optional Torch skips at
  the last completed check).
- Turing raw artifacts: `/scratch/node02/aryama.murthy/novelty-distill/` and
  `/scratch/node03/aryama.murthy/novelty-distill/`, with `logs/`, `evaluations/official/`,
  `checkpoints/`, and staged pinned upstream repositories.

The report is intentionally split into a completed TOMATO section and a live external-run section
so that partial official artifacts cannot be mistaken for final benchmark numbers.
