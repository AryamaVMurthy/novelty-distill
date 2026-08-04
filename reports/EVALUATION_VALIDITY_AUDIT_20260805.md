# Evaluation validity audit and research path

Date: 2026-08-05 (Asia/Kolkata)

## Executive conclusion

The repository contains a useful **systems-complete pilot**, but not yet a
paper-complete evaluation. Six compact 4B LoRA baselines were trained, the
untouched student and 14B teacher controls were generated, 1,658 prompts were
evaluated at matched K=4, and the forward/reverse and static/on-policy
distillation paths execute end to end. The strongest defensible pilot result is
that off-policy forward-KL (`C1-best1`) repairs the severe quality loss of hard
sequence KD (`B2b`) and is not improved by the current on-policy forward-KL
run (`D1`). Reverse-KL methods have similar Qwen-judge quality but less
quality-qualified breadth at the stricter embedding boundaries.

That conclusion must remain narrow. The post-hoc audit found four material
validity failures:

1. the old official NoveltyBench runs accidentally repeated one sampling seed
   for all ten generations and are invalid;
2. TOMATO historical targets contain corpus-level cross-topic passage
   contamination, invalidating every human-target or privileged-target method;
3. TOMATO's issue-month split is not a strict earliest-public temporal split;
4. the old `viable_semantic_yield` gate omitted feasibility, while the 0.94
   embedding boundary is highly uncalibrated and unstable.

The current Qwen3-32B judge is useful for operational feasibility and
soundness diagnostics. It does not evaluate novelty, is from the same model
family as the generator/teacher, has three nearly saturated dimensions, and
was also used to choose the `best1` training targets. It cannot support a
paper-level scientific-quality or novelty claim without an independent judge
and blinded human calibration.

## What was actually built and run

### Data and teacher production

- Training inputs: 1,000 TOMATO scientific prompts.
- Evaluation inputs: 1,658 TOMATO prompts labelled as October 2025 by issue
  month.
- Training teacher: pinned Qwen3-14B, eight prompt-only generations per
  training prompt, producing 8,000 candidates.
- Candidate selector/evaluation judge: pinned Qwen3-32B-FP8.
- Student: pinned Qwen3-4B with rank-16 LoRA adapters.
- Common generation request: one hypothesis and test plan, at most 300 words,
  with a mechanism, distinction from existing work, intervention, measurable
  outcomes, and falsification condition.
- Common student budget: 1,000 prompt/example exposures and seed 17.

The prompt-only Qwen3-14B teacher bank is still usable. It does not inherit the
historical-target corruption because it was generated from the scientific
prompt, not copied from the suspect answer field.

### Baseline ladder

| Method | Training signal | Status after audit |
|---|---|---|
| `A0` | untouched Qwen3-4B | valid operational control |
| `A1` | Qwen3-14B evaluation teacher | valid operational reference |
| `B1` | SFT on historical human target | **quarantined** |
| `B2b` | hard sequence KD on judge-selected best-of-8 teacher answer | usable pilot; selection bias remains |
| `C1-best1` | static-trajectory token-level forward KL | usable pilot; best1 selection bias remains |
| `C2-best1` | static-trajectory token-level reverse KL | usable pilot; best1 selection bias remains |
| `D1` | on-policy GKD forward KL on student trajectories | usable pilot |
| `D2` | on-policy GKD reverse KL on student trajectories | usable pilot |

The other historical-target methods `A3`, `C1-human`, `C2-human`, `E1`, `E2`,
and `E3` are also quarantined. They remain reproducibility and failure-analysis
artifacts, not treatment evidence.

### Evaluation protocol

For each TOMATO evaluation prompt, the teacher reference and each student were
sampled four times. The fixed judge received the complete rendered task and one
candidate answer with no method label. It returned strict JSON ratings from 1
to 5 for relevance, feasibility, soundness, clarity, and instruction
compliance. The old aggregate quality number is the normalized mean of those
five values; it is not a novelty score.

Qwen3-Embedding-4B then represented each idea by its mechanism, intervention,
and experimental test. Teacher samples were complete-link clustered. Student
samples were admitted to a teacher mode only if they met the similarity
boundary against every teacher-mode member; unmatched samples formed separate
student modes. Eight cosine boundaries from 0.70 through 0.95 were retained.

Good aspects of this design are the matched prompts, matched K, frozen model
revisions, prompt-level statistical unit, complete-link rather than chaining
clusters, preserved raw scores, threshold curves, finish-reason diagnostics,
content hashes, and paired analyses. These are worth keeping.

## Data validity findings

### Historical target corruption

The generic 12-gram target-integrity scanner audited all 2,658 train and test
rows. It found 56 suspicious repeated-passage groups spanning 112 unique rows.
The largest group repeats an oxytocin/Oxtr adult-born granule-cell passage in
30 documents; 29 of those prompts do not support that passage. Other repeated
blocks describe unrelated mouse cohorts, water-maze experiments, and reused
statistical procedures. There are no exact full-target duplicates, which is
why a simple duplicate-row check missed the defect.

Every released target also equals its privileged-target field. The mean target
length is 340.77 words, the median is 318, and 1,468/2,658 exceed the stated
300-word response limit. This is not evidence that every target is wrong, but
it is enough to invalidate a treatment whose supervision or privileged context
uses the unreviewed field. The immutable scanner output is
[`tomato-target-integrity-20260805.json`](audits/tomato-target-integrity-20260805.json).

### Temporal split failure

PubMed metadata was fetched for all 2,658 PMIDs and the earliest electronic,
ahead-of-print, or print availability date was retained. Among the 1,658 test
papers, 4 were already public by 2023-12-31, 26 by 2024-12-31, 161 by the
Qwen3 release date of 2025-04-29, and 522 by 2025-09-30. One row labelled
`2025_37032452` was electronically public on 2023-04-09 even though its issue
date is October 2025.

This does **not** prove that Qwen saw any paper in pretraining. It proves only
that issue month cannot establish leakage safety. Qwen's public model card does
not provide a sufficiently precise training-data cutoff, and a model release
date is not a training cutoff. The current split should therefore be called an
issue-month holdout, not strict temporal OOD. The immutable audit is
[`tomato-pubmed-temporal-audit-20260805.json`](audits/tomato-pubmed-temporal-audit-20260805.json).

## Judge audit

Across the untouched A0 K=16 run (26,528 answers), aggregate quality had mean
0.9339, median 0.95, and a 25.4% exact ceiling rate. Dimension-level rating-5
rates were 99.9% for instruction compliance, 95.9% for clarity, 92.6% for
relevance, 64.1% for soundness, and 25.4% for feasibility. The first three are
format checks with little useful headroom. Feasibility and soundness are the
only dimensions currently suitable for primary operational comparisons.

The judge is not completely collapsed: feasibility and soundness vary, and
the aggregate has a mean within-prompt range of 0.130. It is nevertheless not
scientifically calibrated. The central risks are:

- same-family preference: generator, teacher, embedder, and judge are all Qwen;
- target-selection circularity: the same pinned Qwen3-32B judge selected
  `best1` teacher targets and later rated models trained from them;
- rubric saturation: fluent, compliant answers get near-max relevance and
  clarity almost automatically;
- no literature retrieval: the judge cannot determine whether a mechanism is
  already published;
- no uncertainty or rationale audit in the saved five-number schema;
- absolute-score anchoring and possible length/style preference;
- no demonstrated agreement with domain experts.

A previous informal comparison also incorrectly contrasted the 1,000-prompt
training teacher scores with A0 on a different 1,658-prompt set. The corrected
same-prompt K=4 control shows the expected ordering: A1 scores 4.603
feasibility and 4.921 soundness versus A0's 4.218 and 4.603. Cross-prompt judge
means must not be used as model comparisons.

## Corrected K=4 results

Jobs 19055--19062 reused the complete frozen generations and judge records and
recomputed every threshold with a corrected gate: relevance, feasibility,
soundness, and clarity must all be at least 4. No new judge calls or training
were involved.

| Method | Feasibility /5 | Soundness /5 | Corrected yield @0.94 | Yield @0.90 | Teacher recall @0.94 | Teacher recall @0.90 |
|---|---:|---:|---:|---:|---:|---:|
| `A0` | 4.218 | 4.603 | 1.878 | 1.240 | 0.120 | 0.628 |
| `A1` | 4.603 | 4.921 | 1.962 | 1.279 | 1.000 | 1.000 |
| `B1`* | 3.636 | 3.603 | 1.700 | 1.335 | 0.025 | 0.395 |
| `B2b` | 3.398 | 3.176 | 0.460 | 0.384 | 0.031 | 0.419 |
| `C1-best1` | 4.239 | 4.657 | 2.224 | 1.320 | 0.134 | 0.653 |
| `C2-best1` | 4.240 | 4.656 | 1.767 | 1.154 | 0.135 | 0.658 |
| `D1` | 4.234 | 4.658 | 2.197 | 1.302 | 0.117 | 0.640 |
| `D2` | 4.231 | 4.638 | 1.720 | 1.172 | 0.125 | 0.638 |

`B1` is shown only to preserve the complete audit; its training evidence is
quarantined.

The corrected gate lowers old yield most for B2b (0.565 to 0.460) and B1
(1.799 to 1.700). It changes C1, C2, D1, D2, and A0 by only about 0.02--0.03.
Thus the main pilot interpretation survives the bug fix:

- `B2b` is much worse than C1 on both judged quality and qualified breadth;
- C1 and D1 are essentially tied on quality and breadth;
- C2 matches C1 on judged feasibility/soundness but has lower qualified yield;
- D2 similarly has lower qualified yield than D1 and slightly lower soundness;
- A0 is already strong, so any future method must beat a serious untouched
  control rather than only a weak SFT baseline.

The magnitude of every semantic metric is threshold dependent. For A0,
teacher recall falls from 0.628 at 0.90 to 0.120 at 0.94, while corrected yield
rises from 1.240 to 1.878 because a stricter similarity requirement labels more
answers as distinct. C1 exceeds C2 in corrected yield at all eight audited
boundaries, but the gap is small at 0.70--0.82 and large at 0.90--0.95. D1
exceeds D2 at six of eight boundaries, with tiny reversals at the two lowest
boundaries. These are useful sensitivity patterns, not calibrated semantic
effect sizes.

## External benchmark audit

The old NoveltyBench result is withdrawn. Its adapter instantiated the same
generation callable ten times under one global seed; the raw A0 artifact
contains the same dog story ten times. This forced `Distinct@10=1` for every
method and made the utility comparison non-independent.

The repaired adapter now issues ten calls with explicit seeds 17--26,
temperature 1.0, and top-p 1.0, matching NoveltyBench's intended independent
sampling. Corrected smoke job 19053 evaluated two prompts: all 10 raw
completions were unique for both prompts, Distinct@10 was 3.5, and Utility@10
was 3.319. This proves the implementation path only; two prompts cannot compare
models. The frozen artifact is
[`noveltybench-corrected-smoke-A0-k10-seed17-v2/`](audits/noveltybench-corrected-smoke-A0-k10-seed17-v2/).

The historical HypoSpace component files were not affected by that exact
Inspect seed bug, but the combined matrix remains withdrawn because one of its
four components is invalid. A separate artifact audit verified matched task
IDs, corrected LoRA routing, complete K=10 adaptive searches, and zero provider
errors. It also found severe parser confounding and no retained raw text for
1,438 syntactically unparseable causal/3D requests. HypoSpace can therefore be
kept only as a secondary domain-separated structured-search diagnostic. It
measures constrained hypothesis-space recovery/validity, not open-ended
scientific novelty. See
[`HYPOSPACE_VALIDITY_AUDIT_20260805.md`](HYPOSPACE_VALIDITY_AUDIT_20260805.md).

## What the existing statistics can and cannot say

The paired prompt-level analysis and Holm correction are appropriate for the
fixed run. The large C1-versus-B2b feasibility and soundness effects are real
under this Qwen judge. C2-versus-C1 and D1-versus-C1 show no practically useful
quality difference. D2 is about 0.018--0.020 lower in judged soundness than the
other KL controls.

However, every trained method has only one training seed. Thousands of prompts
do not replace independent training replications: they precisely estimate the
behavior of these particular checkpoints, not variance across training. The
K=4 answers are repeated samples nested inside prompts and must never be
treated as independent rows. Semantic p-values at a hand-selected 0.94 boundary
are not valid primary evidence. Nearest-training-target similarity is also not
novelty: taking a maximum over 8,000 stylistically similar teacher answers is
an extreme-value similarity diagnostic.

## Relation to the literature

The baseline ladder itself is standard. Hard teacher-output SFT corresponds to
[sequence-level KD](https://aclanthology.org/D16-1139/). Static and on-policy
token-level KL controls follow the design space formalized by
[GKD](https://arxiv.org/abs/2306.13649); GKD specifically motivates student
trajectories to reduce train/inference distribution mismatch. Reverse KL is a
standard generative-distillation control, including
[MiniLLM](https://arxiv.org/abs/2306.08543), but mode seeking is an empirical
finite-training question rather than a theorem that this experiment can assume.

For idea evaluation, the strongest directly relevant precedent is the ICLR
2025 human study
[Can LLMs Generate Novel Research Ideas?](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea94957d81b1c1caf87ef5319fa6b467-Abstract-Conference.html).
It used anonymized ideas, balanced assignments, 2--4 expert reviews per idea,
and 298 reviews. Even its best tested LLM evaluator agreed with the expert
top/bottom ranking less than the paper's human split-half consistency. That is
strong evidence against treating our one absolute Qwen rating as ground truth.

[NoveltyBench](https://arxiv.org/abs/2504.05228) is a useful domain-general
response-diversity transfer test, but its learned equivalence classifier was
trained for functional equivalence on generic prompts; it is not a scientific
novelty benchmark. The user's July 2026 paper,
[Measuring the Gap Between Human and LLM Research Ideas](https://arxiv.org/pdf/2607.01233),
is valuable for secondary taxonomy analysis: opportunity patterns, method
paradigms, TVD/JSD, and normalized entropy. It calibrated the automatic
taxonomy on an author-audited subset. We can reuse the *analysis idea*, with
attribution and our own human calibration, but its labels should not become a
replacement outcome for feasibility or semantic equivalence.

The verified bibliography and claim-by-claim research log are
[`LITERATURE_SOURCE_LOG_20260805.md`](LITERATURE_SOURCE_LOG_20260805.md) and
[`../references.bib`](../references.bib).

## Recommended ICLR research question

Do not frame the current paper as “we improve novelty.” A defensible and useful
question is:

> How do supervision type, KL direction, and trajectory source change the
> quality--breadth trade-off when distilling open-ended scientific ideation from
> a 14B teacher into a 4B student?

That claim matches what the experiments actually manipulate and avoids asking
an ungrounded LLM judge to certify novelty.

## Minimum paper-grade experiment

1. Repair the corpus contract. Exclude all historical-target supervision until
   the source dataset is regenerated or manually reviewed. Rebuild the test
   split by earliest public date relative to a documented model cutoff, or call
   it an issue-month holdout and report cutoff strata explicitly.
2. Keep a compact core ladder: A0, random-1 sequence KD, best-1 sequence KD,
   C1, C2, D1, and D2. Random-1 is essential because best-of-8 selection is
   currently confounded with the same judge later used for evaluation.
3. Train at least three independent seeds, 17/29/43, at the same 1,000
   exposures. Report checkpoint-level variance and a hierarchical analysis
   with prompts nested inside checkpoints.
4. Calibrate semantic equivalence before future test comparison. Draw response
   pairs around the full similarity range from a method-independent calibration
   source, blind them, collect at least two domain-aware labels per pair,
   adjudicate disagreements, and freeze the threshold before future multi-seed
   test outputs. The prepared packet uses clean prompt-only teacher pairs from
   the training bank and therefore contains no held-out method identity.
5. Independently calibrate quality. Use a different-family LLM judge for cheap
   screening, then a blinded human subset for feasibility and soundness. Report
   weighted agreement, rank correlation, ceiling rates, per-method rank
   stability, and failure examples. LLM--LLM agreement is a sensitivity check,
   not human validity.
6. Make feasibility and soundness the primary outcomes. Keep aggregate quality,
   teacher-mode precision/recall, ClusterJSD, and corrected qualified yield as
   separately named secondary outcomes. Preserve every threshold curve.
7. Run the corrected full NoveltyBench 100-prompt × K=10 protocol only after
   the A0 full-run gate passes. Treat it as generic diversity transfer. Audit
   and report HypoSpace domains separately.
8. Add one robustness axis after the main result is stable: either a 1.7B
   student, an 8B teacher, or an independent prompt domain. Do not launch all
   scale combinations before three-seed 4B results establish a real effect.
9. Freeze hypotheses, exclusions, primary metrics, and analysis code before
   training the proposed new method. Compare that method first to C1, then to
   C2/D1/D2 as mechanism controls.
10. Release exact prompts, target-construction rules, raw judge records,
    calibration packets, response hashes, model revisions, job manifests, and
    negative findings. Clearly distinguish checkpoint evidence, automatic
    proxy evidence, human evidence, and claims.

## Semantic-equivalence calibration status

The no-inference preparation stage is complete. The frozen Qwen embedding cache
contained every one of the 28,000 within-prompt pairs from the clean,
prompt-only 1,000-prompt teacher bank. Packet v2 selects 256 pairs from 256
different prompts: 32 in each of eight similarity strata from 0.74 through
1.00. Each rater receives 26 hidden reversed-order repeats, for 282 judgments
per rater. Similarities, prompt IDs, source indices, repeat flags, and model
metadata are absent from the public packet.

The first prepared packet was rejected before human use because a list-filter
implementation allowed multiple pairs from one prompt within a stratum. It had
only 240 unique prompts. A regression test now covers this case; the v2 runtime
gate directly verified 256 originals, 256 unique prompts, 32 pairs per stratum,
and 26 repeats. Its public packet SHA-256 is
`aae74a1f789f8cf2bfadd5fc4f6b6ae95c9a2a238c06b9d7daf6209e25150f31`;
full redacted provenance is in
[`audits/semantic-equivalence-calibration-20260805-v2.manifest.json`](audits/semantic-equivalence-calibration-20260805-v2.manifest.json).

Two domain-aware humans must now label it independently. Every disagreement or
`uncertain` label requires adjudication. The frozen analyzer reports inter- and
intra-rater reliability and selects among thresholds 0.80--0.99 by maximum
balanced accuracy, breaking exact ties toward the higher, more conservative
boundary. This calibrates semantic equivalence only; it does not validate a
scientific-novelty claim.

## DeepInfra calibration plan

DeepInfra is useful here for speed and model-family separation, not as an
automatic truth oracle. The currently available
`meta-llama/Llama-3.3-70B-Instruct-Turbo` is a suitable first sensitivity judge
because it is Llama-family rather than Qwen-family and supports strict
structured output. A cost-controlled first pass should score roughly 420
blinded answers: 60 each from A0, A1, B2b, C1, C2, D1, and D2, stratified over
pooled Qwen feasibility/soundness rank quartiles. The same 60 prompt/sample
slots are used for every method, preserving paired comparisons. Re-score a 10%
subset to measure API/model stability. Expand only if ceiling rate, agreement,
and method ranking are informative.

The first packet was retired before any paid request. It had 420 original
answers plus 42 hidden repeats, but its 60 paired sample slots represented only
58 unique prompts, and it omitted the random-1 SeqKD control. The sampler now
enforces one sample slot per prompt with a regression test. After the matched
K=4 `B2a` scores finish, a replacement packet will contain A0, A1, B2a, B2b,
C1, C2, D1, and D2: 480 originals plus 48 hidden repeats, 528 total calls, with
60 unique paired prompts. The retired packet and reason are preserved in
[`audits/deepinfra-calibration-packet-20260805.SUPERSEDED.md`](audits/deepinfra-calibration-packet-20260805.SUPERSEDED.md).
The runner is resumable, stores raw responses and token usage, validates the
strict schema, and never sends method labels, Qwen scores, or repeat flags to
the external judge. Zero paid requests have been sent so far.

The credential is not currently present as `DEEPINFRA_API_KEY` on the local or
Turing environment. The key pasted into chat should be rotated because it is
exposed, then supplied through the environment rather than committed, written
to a report, or passed on a command line. Until then, all corrected local and
cluster analyses can proceed without spending DeepInfra credits.

## Immediate decision

Use `C1-best1` as the current practical distillation reference, `B2b` as the
hard sequence-KD failure baseline, `C2-best1` as the static reverse-KL control,
and D1/D2 as trajectory-source/KL controls. Do not use B1, the old
NoveltyBench table, a single 0.94 semantic number, or aggregate Qwen quality as
paper evidence. The next compute should be three-seed replication and
calibration, not a larger unstructured method matrix.
