# Baseline distillation study: end-to-end report and validity status

Date: 2026-08-05 (Asia/Kolkata)

Status: seed-17 baseline evidence and the different-family judge audit are
complete. Seed-29/43 replication is running under frozen contracts. This file
will receive the final checkpoint-seed table when the dependency-gated matrix
finishes.

## Executive summary

This project is now a focused **baseline distillation study**, not a novelty
claim. It asks how supervision type, KL direction, and static versus on-policy
trajectories affect a Qwen3-4B student's ability to produce feasible, sound,
and non-collapsed scientific hypotheses from a Qwen3-14B teacher.

The defensible seed-17 result is:

1. The untouched 4B model (`A0`) is already strong.
2. Hard single-output sequence KD fails badly (`B2a`, `B2b`), regardless of
   whether the target is random-1 or Qwen-judge-selected best-1.
3. Token-distribution forward KL (`C1`) repairs that failure and is the main
   trained baseline to beat.
4. Reverse KL (`C2`) preserves Qwen-judged feasibility/soundness but produces
   less embedding-qualified breadth at strict thresholds.
5. Current on-policy forward/reverse KL (`D1`, `D2`) does not improve quality
   over the corresponding static KL controls.
6. A different-family Llama judge independently confirms the large hard-KD
   soundness failure, but is too generous and compressed to validate absolute
   quality. It does not measure novelty.
7. The semantic boundary is highly threshold-sensitive and remains
   uncalibrated until two humans label the frozen equivalence packet.

The completed three-seed B2a/B2b replication now strengthens item 2: both
random-1 and best-1 hard KD remain far below A0, while best-1 minus random-1
changes only +0.0216 feasibility and +0.0235 soundness on average. The
three-seed C1/C2 replication also supports items 3--4: their judged quality
is tied, while C2 has lower strict-threshold embedding-defined breadth. The
The three-seed D1 result is also a quality tie with C1 and shows no on-policy
forward-KL advantage. D2's conclusion remains pending until its last active
seed-43 checkpoint evaluation finishes.

The correct research use of these results is to select controls for a later
new method: A0, B2a/B2b, C1, C2, D1, and D2. The new method should first beat
C1 on a preregistered quality--breadth trade-off without reducing feasibility
or soundness.

## What happens end to end

| Stage | Input | Operation | Output |
|---|---|---|---|
| Prompt preparation | 1,000 TOMATO training rows | remove answer/privileged fields; retain task context | clean prompt-only training set |
| Teacher production | each clean prompt | pinned Qwen3-14B generates eight answers | 8,000 prompt-only teacher candidates |
| Teacher scoring | candidate plus complete task | pinned Qwen3-32B-FP8 scores five rubric dimensions | random-1 and best-1 target views plus raw scores |
| Student training | same 1,000 prompts/exposures | SFT, static GKD, or on-policy GKD | rank-16 Qwen3-4B LoRA adapter |
| Held-out generation | 1,658 issue-month holdout prompts | each model generates K=4 answers under matched decoding | 6,632 answers per checkpoint |
| Operational quality | task plus one blinded answer | Qwen3-32B-FP8 scores feasibility/soundness and format axes | per-answer five-point scores |
| Semantic diagnostics | mechanism/intervention/test representations | Qwen3-Embedding-4B plus complete-link clustering | threshold curves, mode recall/precision, qualified yield |
| External transfer | 100 generic NoveltyBench prompts | ten independent generations, seeds 17--26 | functional Distinct@10 and Utility@10 |
| Judge audit | 60 paired seed-17 slots per method plus repeats | DeepInfra Llama-3.3-70B independently scores blinded answers | agreement, compression, repeat reliability, paired sensitivity |
| Replication | training seeds 17, 29, 43 | repeat six clean trained methods | checkpoint-seed uncertainty rather than prompt pseudoreplication |

## Data actually used

### Training

- Dataset slice: 1,000 prompt-only TOMATO rows.
- Domain: the configured TOMATO/PubMed slice is predominantly biomedical,
  clinical, neuroscience, and life-science research. This experiment is not a
  general cross-discipline study.
- Teacher bank: eight Qwen3-14B generations per prompt, 8,000 total.
- Teacher production finish reasons: 7,985 `stop`, 15 `length`; length-stop
  rate 0.1875%.
- Clean methods use only the scientific prompt and prompt-only teacher
  generations. They do not use the released historical answer field.
- All trained methods consume 1,000 prompt/example exposures. Four-target
  exploratory pools are not part of the clean core ladder.

A complete illustrative record example is
[`examples/tomato_record_2022_36254448.json`](../examples/tomato_record_2022_36254448.json).
It shows the question/background/inspiration and withheld target structure;
the clean training pipeline exposes only the prompt-side fields to the methods
in this report. `scripts/preview_tomato.py` provides the repository's local
browser/terminal preview path without dumping the whole dataset at once.

### Held-out evaluation

- 1,658 TOMATO rows labelled as October 2025 by journal issue month.
- K=4 matched generations per prompt/checkpoint.
- The split must be called an **issue-month holdout**, not strict temporal OOD.
  Earliest-public PubMed metadata shows that 522/1,658 test papers were public
  by 2025-09-30, including 161 before Qwen3's public release. Release date is
  not a documented training cutoff, so model leakage is unknown rather than
  proven or excluded.

### Data excluded from claims

The historical/privileged target field is quarantined. A corpus-level 12-gram
audit found 56 suspicious repeated-passage groups across 112 rows, including
large passages copied across unrelated topics. It also found 1,468/2,658
targets over the 300-word response contract. B1 and every method trained from
or conditioned on that field are retained only as failure/provenance artifacts.

## Models and baseline ladder

| ID | Role | Training signal | Status |
|---|---|---|---|
| A0 | untouched Qwen3-4B | none | primary base control |
| A1 | Qwen3-14B teacher | none | teacher reference, not a student |
| B2a | random-1 sequence KD | cross-entropy on one random teacher answer | hard-KD control |
| B2b | best-1 sequence KD | cross-entropy on Qwen-judge-selected teacher answer | hard-KD plus selection control |
| C1-best1 | off-policy forward KL | teacher/student token distributions on static best-1 trajectories | main trained baseline |
| C2-best1 | off-policy reverse KL | reverse direction on the same static trajectories | static mode-seeking control |
| D1 | on-policy forward KL | teacher distribution on live student trajectories | trajectory-source control |
| D2 | on-policy reverse KL | reverse KL on live student trajectories | on-policy mode-seeking control |
| B1 | historical-target SFT | corrupted released answer field | quarantined |

The distinction between static and on-policy is important. C1/C2 see fixed
teacher-generated trajectories. D1/D2 sample the current student during
training, then compare teacher and student token distributions on those live
trajectories. D1 and D2 were therefore genuinely run as on-policy GKD, not
ordinary SFT with a different label.

## Frozen training contract

- Student: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Teacher: `Qwen/Qwen3-14B`, revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- LoRA: rank 16, alpha 32.
- Optimizer exposure: 125 steps × batch 1 × gradient accumulation 8 = 1,000
  examples/prompt exposures.
- Learning rate: 2e-5.
- Seeds: 17, 29, 43.
- Checkpoints: every 25 steps; downstream generation requires the validated
  final adapter, never an intermediate checkpoint.
- On-policy decoding: temperature 0.7, top-p 0.8, top-k 20, maximum 512 new
  tokens, non-thinking interface.
- Held-out decoding: the same temperature/top-p/top-k controls, K=4, maximum
  512 new tokens, and one answer capped by instruction at 300 words.

## What the model sees

Every generator receives the complete task text: research question,
background, and supplied inspiration where present. The response instruction
asks for one hypothesis and test plan in at most 300 words, including a
specific central mechanism, difference from existing approaches, intervention
or experiment, measurable outcomes, and a falsification condition. It must not
repeat the background or add a preamble.

The Qwen quality judge receives only:

```text
Task:
<complete rendered scientific task>

Candidate answer:
<one generated hypothesis and test plan>
```

It does not receive the method label. Its deterministic rubric separately
defines relevance, feasibility, soundness, clarity, and instruction
compliance on a 1--5 scale and explicitly says not to assign 5 merely for
fluent or long prose. The normalized five-axis mean is a quality proxy, not a
novelty score.

## Evaluation metrics and how to read them

### Primary operational axes

- **Feasibility /5:** whether the proposed test is actionable with credible
  measurements and resources.
- **Soundness /5:** whether the mechanism is coherent and the claimed
  conclusion is supported by the proposed evidence.

These are the only automatic Qwen dimensions with enough variation for the
core baseline comparison. They remain judge-relative, not human truth.

### Saturated diagnostics

- Relevance, clarity, and instruction compliance mostly measure whether a
  fluent answer followed the format. Their high ceiling rates leave little
  model-separation headroom.
- Composite quality averages those dimensions and can hide a feasibility or
  soundness defect. It is secondary.
- Length-stop rate is the fraction that reaches the 512-token cap. It detects
  truncation; it is not a quality or diversity outcome.

### Semantic/embedding diagnostics

- Teacher-mode recall/precision compare student modes to four sampled teacher
  answers.
- Student semantic clusters and ClusterJSD summarize embedding geometry.
- Quality-qualified semantic yield counts distinct student modes only after
  each answer passes relevance, feasibility, soundness, and clarity ≥4.

These are threshold curves, not calibrated novelty. At similarity 0.90 A0
teacher recall is 0.628; at 0.94 it is 0.120. A stricter threshold can
simultaneously lower recall and increase “distinct” yield, so the 0.94 number
must never be interpreted alone. The frozen two-human equivalence packet is
the required boundary-selection gate.

### External transfer

Corrected NoveltyBench tests generic functional response diversity and utility
with ten independent generations. It is useful for detecting broad response
collapse, but it is not a scientific novelty benchmark. HypoSpace is retained
only as a domain-separated constrained-search diagnostic because parser
failures confound several historical results.

## Corrected seed-17 TOMATO results

| Method | Feasibility /5 | Soundness /5 | Qualified yield @0.94* | Teacher recall @0.94* | Length stop |
|---|---:|---:|---:|---:|---:|
| A0 | 4.2176 | 4.6025 | 1.8776 | 0.1204 | 0.0000% |
| A1 | 4.6028 | 4.9213 | 1.9620 | 1.0000 | 0.0000% |
| B2a | 3.3976 | 3.1654 | 0.4530 | 0.0251 | 0.0000% |
| B2b | 3.3984 | 3.1761 | 0.4602 | 0.0306 | 0.0000% |
| C1-best1 | 4.2385 | 4.6574 | 2.2238 | 0.1336 | 0.0000% |
| C2-best1 | 4.2399 | 4.6558 | 1.7672 | 0.1348 | 0.0000% |
| D1 | 4.2336 | 4.6583 | 2.1972 | 0.1173 | 0.0000% |
| D2 | 4.2307 | 4.6380 | 1.7195 | 0.1254 | 0.0151% |

`*` Threshold-dependent descriptive metrics pending human equivalence
calibration.

Interpretation:

- B2a versus B2b is a near-exact quality tie. Only 110/1,000 training targets
  overlap, and held-out exact output matches are rare, so the shared failure is
  hard single-output imitation rather than duplicate artifacts or best-of-eight
  selection alone.
- C1 is +0.841 feasibility and +1.492 soundness over B2a on paired held-out
  prompts. It repairs the hard-KD collapse.
- C1/C2/D1 are essentially tied in judged quality. D1 does not show an
  on-policy advantage.
- D2 is also high-quality but has slightly lower Qwen soundness and lower
  strict-threshold qualified yield than D1/C1.
- A0 remains competitive with every KL student. A new method must beat A0 and
  C1, not only the weak hard-KD controls.

## Different-family judge audit

DeepInfra Llama-3.3-70B scored 480 blinded originals plus 48 hidden repeats.
All 528 calls passed strict parsing, ended with finish reason `stop`, and cost
USD 0.062097 in the accepted run.

It independently reproduces the large hard-KD soundness failure:

- C1 minus B2a: +0.683, paired 95% bootstrap interval [0.533, 0.817].
- C1 minus B2b: +0.700 [0.567, 0.833].
- B2b minus B2a: -0.017 [-0.117, 0.083].
- D1 minus C1: 0.000 [-0.067, 0.067] soundness.
- D2 minus C2: 0.000 [-0.067, 0.067] soundness.

But it confirms the judge-validity concern rather than solving it:

- 479/480 clarity ratings are 5.
- 446/480 feasibility ratings are exactly 4.
- 365/480 soundness ratings are 5.
- 0/480 answers receive a fatal flaw.
- Feasibility Spearman agreement with Qwen is 0.231; soundness is 0.583.
- 54.6% of rationales are exact duplicates.

Post-run examples show Llama assigning feasibility/soundness 4/4 to an
undefined “controlled population” lead-exposure experiment and to an ICA study
whose proposed randomization does not identify the requested subgroup
comparison. High repeat consistency therefore reflects deterministic behavior,
not expert validity. The complete audit is
[`DEEPINFRA_INDEPENDENT_JUDGE_FINDINGS_20260805.md`](DEEPINFRA_INDEPENDENT_JUDGE_FINDINGS_20260805.md).

### Remaining automatic-evaluation risks

- Qwen generator, teacher, embedder, and main judge are from one model family,
  so shared preferences can inflate agreement.
- Qwen3-32B selected the best-1 teacher targets and later judged students
  trained from those targets. B2b/C1/C2 therefore retain a selection/judging
  circularity even though method labels are hidden at evaluation.
- Neither automatic judge retrieves or compares the current literature, so
  neither can certify global novelty or rediscovery.
- Both judges reward fluent structure; Llama's qualitative disagreements show
  that headings and a nominal test plan can mask identification, ethics, or
  mechanism-test mismatch.
- Embedding similarity comes from another Qwen model and changes semantic
  conclusions sharply with the threshold. It measures representation geometry,
  not expert equivalence until calibrated.
- The evaluation prompts are mostly biomedical. A single judge can lack the
  specialized expertise needed across every included subfield.

## Corrected NoveltyBench transfer

The original matrix is withdrawn because all ten generations accidentally
shared one seed. The corrected runs use explicit seeds 17--26 for every prompt.

| Method | Distinct@10 | Utility@10 |
|---|---:|---:|
| A0 | 4.080 | 4.066 |
| B2a | 5.320 | 4.720 |
| B2b | 5.520 | 4.956 |
| C1-best1 | 5.580 | 5.152 |
| C2-best1 | 4.600 | 4.624 |
| D1 | 5.500 | 5.121 |
| D2 | 4.510 | 4.507 |

C1 and D1 are prompt-paired ties; C2 and D2 are also ties. Every trained model
improves the generic aggregate over A0 at seed 17, including B2a/B2b despite
their severe TOMATO soundness failure. That disagreement is useful evidence
that generic functional diversity and scientific-idea quality are different
constructs.

## Three-seed replication status

The clean six-method matrix is frozen at seeds 17/29/43. Offline seed-29/43
B2a/B2b/C1/C2 adapters are complete and hash-staged. All four D1/D2 on-policy
replications have now completed step 125 with the identical 1,000-exposure
budget. Every final adapter is 132,187,888 bytes and byte-identical to its
checkpoint-125 adapter; every trainer state has global step 125, epoch 1.0,
and exactly 125 finite loss/gradient records. Direct safetensors-header audits
find 504 non-empty F32 tensors in every final adapter. The ordered 1,000-row
training-ID fingerprint is identical across all four runs:
`190e3ea7cb48a13dbd5a142f3aa85ff5f355fd69ffd89bf8ee75a47c9d3aa7dc`.
The final adapter SHA-256 values are
`6b69301f105c915d5845973394075bd88111f2b7c4dd523a055ec56b7043069f`
(seed-29 D1),
`42580b90dd9a679e515016f607ac3fc6ddaed45a76e7c430777be1aa2f42dd57`
(seed-29 D2),
`8889c3873d1b22a5642909f7395d220f6de41ea13a726f5f20514d6d9d36f28d`
(seed-43 D1), and
`09e4cee5f80e88ce1f3404c5e19042c6c8b1d96b8da3cc08384aec347631b6c4`
(seed-43 D2). Cross-node staging jobs copied seed-29 D1 and seed-43 D2
byte-identically and released
their held evaluation arrays only after complete-tree identity checks passed.
Held-out D1/D2 evaluation is now running from these final adapters; no
intermediate checkpoint is an evaluation model.

The first eleven new held-out checkpoint results are complete. Seed-29 B2a
produced feasibility 3.3602 and soundness 3.1530 over all 1,658 prompts after
its global generation and score gates passed. Its corrected evaluation
artifact is 13,276,716 bytes with SHA-256
`8cd184a5aa40e7a9f847e0a1394d9d1f8db2cbb8da42c93127c3910c6d526e90`.
This closely matches seed 17 (3.3976/3.1654) and is early replication evidence
for the hard-KD failure, but it is not the declared three-seed conclusion.
A checksum-matched local backup is retained at
[`artifacts/compact-k4-three-seed-raw/B2a-tomato1k-seed29-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/B2a-tomato1k-seed29-temporal-k4-corrected-v2.json);
the directory is intentionally gitignored because it will contain the complete
raw replication matrix.

Seed-29 B2b produced feasibility 3.4095 and soundness 3.1975 over the same
1,658 prompts. This also closely matches seed 17 (3.3984/3.1761). Within seed
29, best-1 selection improves only +0.0493 feasibility and +0.0445 soundness
over random-1 B2a, so the second seed continues to suggest that target
selection alone does not repair hard single-output sequence KD. This paired
checkpoint comparison remains preliminary until seed 43 completes. The B2b
artifact is 13,284,916 bytes with SHA-256
`b03bbc6c32e83ef32f2d84319f3aa09414c8ab2f599843479196e944b3d78a65`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/B2b-tomato1k-seed29-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/B2b-tomato1k-seed29-temporal-k4-corrected-v2.json).

Seed-29 C1 is also complete: feasibility is 4.2405 and soundness is 4.6592,
nearly identical to seed 17 (4.2385/4.6574). Within seed 29, forward KL is
+0.8803 feasibility and +1.5062 soundness above random-1 B2a, and +0.8310
feasibility and +1.4617 soundness above best-1 B2b. This is strong second-seed
replication of the hard-KD recovery, while the three-seed interval remains
pending. Its descriptive strict-threshold qualified yield is 2.1785, but that
quantity remains quarantined until human boundary calibration. The C1 raw
artifact is 13,274,613 bytes with SHA-256
`ac2db25f89ec3b6bd8c58aac4ac71fca479fcad87da771b07d39c082e7a1ad1a`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/C1-best1-tomato1k-seed29-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/C1-best1-tomato1k-seed29-temporal-k4-corrected-v2.json).

Seed-29 C2 is complete as well: feasibility is 4.2387 and soundness is
4.6555, only -0.0018 and -0.0038 below the seed-matched C1 checkpoint. Its
strict-threshold qualified yield is 1.7413, -0.4373 below C1, while both
checkpoints have identical teacher-mode recall (0.134449). This reproduces the
seed-17 pattern of a forward/reverse-KL operational-quality tie with lower
embedding-defined breadth under reverse KL. The breadth comparison remains
descriptive until the frozen two-human semantic-boundary calibration is
complete. The 13,273,760-byte raw artifact has SHA-256
`41b697004e419df85d9f1221816e8bb4f263d3619cbd1380120bfa460dcd6147`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/C2-best1-tomato1k-seed29-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/C2-best1-tomato1k-seed29-temporal-k4-corrected-v2.json).

Seed-43 B2a is complete: feasibility is 3.3979, soundness is 3.1800,
strict-threshold qualified yield is 0.4686, and teacher-mode recall is 0.0255.
The clean B2a random-1 SeqKD control is therefore complete over all three
checkpoint seeds. Its across-seed feasibility/soundness means are
3.3853/3.1662. Relative to the frozen A0 realization, the mean checkpoint
effects are -0.8323 feasibility (seed SD 0.0217; descriptive 95% t interval
[-0.8862, -0.7785]) and -1.4364 soundness (seed SD 0.0135; interval
[-1.4699, -1.4028]). This confirms that the hard-KD failure is stable across
the three independently trained checkpoints. Only one of the 19,896 B2a
held-out generations across the three checkpoints ended by length limit, so
the collapse is not an output-cap artifact. As declared below, these
intervals condition on one fixed A0 generation realization and do not include
A0 sampling uncertainty. The 13,283,421-byte raw artifact has SHA-256
`5cc9714fdc190b023a3c7634a3c3d3474bf68c4935f0877b065b285aabc82f81`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/B2a-tomato1k-seed43-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/B2a-tomato1k-seed43-temporal-k4-corrected-v2.json).

Seed-43 B2b is complete as well: feasibility is 3.4125, soundness is
3.1954, strict-threshold qualified yield is 0.4885, and teacher-mode recall
is 0.02694. This completes the best-1 SeqKD control over all three checkpoint
seeds. Its across-seed feasibility/soundness means are 3.4068/3.1897.
Relative to the frozen A0 realization, the mean checkpoint effects are
-0.8108 feasibility (seed SD 0.0075; descriptive 95% t interval
[-0.8293, -0.7922]) and -1.4128 soundness (seed SD 0.0118; interval
[-1.4422, -1.3835]). Relative to seed-matched random-1 B2a, best-1 selection
changes feasibility by only +0.0216 on average (seed SD 0.0250; interval
[-0.0406, 0.0837]) and soundness by +0.0235 (seed SD 0.0183; interval
[-0.0219, 0.0690]). Thus judge selection of one teacher answer does not
repair the hard single-target KD failure at this replication scale. Seed 43
had zero length-stopped B2b outputs. The 13,282,765-byte raw artifact has
SHA-256
`6989ed29df85943fa2e8a5ea542f84d969c20722cc99d2b878114eb9cfe7427d`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/B2b-tomato1k-seed43-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/B2b-tomato1k-seed43-temporal-k4-corrected-v2.json).

Seed-43 C1 is complete: feasibility is 4.2550, soundness is 4.6710,
strict-threshold qualified yield is 2.1671, teacher-mode recall is 0.13586,
and one of 6,632 outputs ended by length limit. This completes static forward
KL over all three checkpoint seeds. Its across-seed feasibility/soundness
means are 4.2447/4.6625. Relative to seed-matched B2b, C1 improves feasibility
by +0.8379 on average (seed SD 0.0061; descriptive df=2 t interval
[0.8228, 0.8529]) and soundness by +1.4729 (seed SD 0.0101; interval
[1.4478, 1.4979]). The recovery from hard single-output KD is therefore
highly stable across these checkpoints.

Relative to the one frozen A0 K=4 realization, C1's mean checkpoint point
estimates are +0.0271 feasibility and +0.0600 soundness. Their
checkpoint-variation-only intervals exclude zero, but they reuse the same A0
sample for all three comparisons and omit A0 generation-sampling uncertainty.
They therefore do not establish that C1 generally outperforms the untouched
model; the defensible claim is that it is at least competitive under the
frozen operational judge. The 13,274,447-byte raw artifact has SHA-256
`20935d1bd974c55e54c18e527865e253c7bf905a4486526ad02df2ef4bc444f8`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/C1-best1-tomato1k-seed43-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/C1-best1-tomato1k-seed43-temporal-k4-corrected-v2.json).

Seed-43 C2 is complete: feasibility is 4.2376, soundness is 4.6526,
strict-threshold qualified yield is 1.7563, teacher-mode recall is 0.14133,
and none of the 6,632 outputs ended by length limit. This completes static
reverse KL over all three checkpoint seeds. Its across-seed
feasibility/soundness means are 4.2387/4.6546. Relative to seed-matched C1,
the mean changes are -0.0059 feasibility (seed SD 0.0100; descriptive df=2 t
interval [-0.0308, 0.0189]) and -0.0079 soundness (seed SD 0.0091; interval
[-0.0306, 0.0147]). Thus forward and reverse static KL are operational-quality
ties at this replication scale.

C2's strict-threshold qualified yield is 1.7549 across seeds, -0.4349 below
C1 (seed SD 0.0230; descriptive interval [-0.4920, -0.3777]). This is a stable
embedding-space pattern, not a calibrated semantic-diversity conclusion: the
threshold metric remains quarantined until the frozen two-human equivalence
study is complete. The 13,271,349-byte raw artifact has SHA-256
`059835a112662534002f2a7d04c4ac31ab51f2f3da29239b7b43c2ff556b7e19`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/C2-best1-tomato1k-seed43-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/C2-best1-tomato1k-seed43-temporal-k4-corrected-v2.json).

Seed-29 D2 is complete: feasibility is 4.2262, soundness is 4.6384,
strict-threshold qualified yield is 1.7310, teacher-mode recall is 0.12485,
and two of 6,632 outputs ended by length limit. Relative to the seed-matched
static reverse-KL C2 checkpoint, the changes are -0.0125 feasibility, -0.0170
soundness, -0.0103 qualified yield, and -0.00960 recall. This second
checkpoint remains consistent with no large benefit from the current
on-policy reverse-KL recipe, but the declared three-seed conclusion still
awaits seed 43. The 13,273,929-byte raw artifact has SHA-256
`2158a02b3017e9f17385a57683f2f59b46a52c09afc15930f2c6e9a0be59bd7e`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/D2-tomato1k-seed29-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/D2-tomato1k-seed29-temporal-k4-corrected-v2.json).

Seed-43 D1 is complete: feasibility is 4.2520, soundness is 4.6755,
strict-threshold qualified yield is 2.1381, teacher-mode recall is 0.10867,
and one of 6,632 outputs ended by length limit. Relative to seed-matched
static forward-KL C1, the changes are -0.0030 feasibility, +0.0045 soundness,
-0.0290 qualified yield, and -0.02719 recall. This checkpoint therefore
repeats the seed-17 pattern of an operational-quality tie without an
on-policy gain; D1 seed 29 remains necessary for the declared three-seed
contrast. The 13,272,191-byte artifact has SHA-256
`7bb7806cbd50dc88a1ccac98b381404b7f6d453a5f82f348ea919fade51331e3`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/D1-tomato1k-seed43-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/D1-tomato1k-seed43-temporal-k4-corrected-v2.json).

Seed-29 D1 is complete: feasibility is 4.2316, soundness is 4.6631,
strict-threshold qualified yield is 2.1677, teacher-mode recall is 0.11811,
and two of 6,632 outputs ended by length limit. D1 is therefore complete over
all three checkpoint seeds. Its across-seed feasibility/soundness means are
4.2390/4.6657. Relative to seed-matched static C1, the mean changes are
-0.0056 feasibility (seed SD 0.0030; descriptive df=2 t interval [-0.0131,
0.0018]) and +0.0031 soundness (seed SD 0.0019; interval [-0.0017, 0.0079]).
Current on-policy forward KL is thus an operational-quality tie with static
forward KL and provides no observed gain.

D1's mean strict-threshold qualified-yield change versus C1 is -0.0221
(interval [-0.0465, 0.0023]), while teacher-mode recall is -0.0200 (interval
[-0.0355, -0.0044]). The recall pattern is stable in the present embedding
geometry but remains a quarantined semantic diagnostic until human
calibration. The 13,277,773-byte artifact has SHA-256
`95c3bc81d007555aca7ed1b232e23d47a057c549539fac5482e8b26c053f0f4b`,
with a checksum-matched local backup at
[`artifacts/compact-k4-three-seed-raw/D1-tomato1k-seed29-temporal-k4-corrected-v2.json`](../artifacts/compact-k4-three-seed-raw/D1-tomato1k-seed29-temporal-k4-corrected-v2.json).

The analysis unit is the independently trained checkpoint seed. Within each
seed, 1,658 prompts are paired and bootstrapped; then the three checkpoint
effects are summarized with an across-seed mean, seed SD, and df=2 t interval.
With only three seeds these intervals will be low-powered and descriptive, but
they avoid the false claim that thousands of prompts replace independent
training replications.

Contrasts between two trained methods pair checkpoints with the same seed.
`B2a - A0` is different: A0 is an untrained fixed control, so the same frozen
A0 K=4 artifact is reused for all three B2a checkpoints. Its across-seed
interval therefore measures variation in the trained B2a checkpoint while
conditioning on one A0 generation realization; it does not include A0
sampling uncertainty.

The scheduler recovery, cross-node artifact hashes, evaluator OOM diagnosis,
replacement jobs, and fail-closed release gates are in
[`audits/compact-three-seed-scheduling-amendment-20260805.md`](audits/compact-three-seed-scheduling-amendment-20260805.md).

## What is valid now

Supported as operational baseline evidence:

- hard single-target SeqKD sharply degrades scientific soundness;
- random-1 and best-1 SeqKD have the same failure across checkpoint seeds
  17, 29, and 43; best-of-eight selection does not repair it;
- token-level KL training avoids that collapse;
- current on-policy GKD has no observed quality advantage over static GKD;
- reverse-KL variants show lower strict-threshold breadth than forward-KL
  variants under the current embedding geometry;
- generic NoveltyBench diversity does not substitute for TOMATO quality.

Not supported:

- any claim that a model's ideas are globally novel;
- any human-quality claim from Qwen/Llama agreement;
- a single-threshold semantic effect before human equivalence calibration;
- strict temporal generalization;
- evidence from B1 or any historical/privileged-target method;
- paper-level seed stability until seeds 29/43 finish.

## ICLR research path

The baseline suite is sufficient groundwork for a later method paper, but is
not itself a top-level novelty contribution. The next method should have one
explicit mechanism, for example preserving multiple teacher modes while using
token-distribution supervision, and a preregistered success condition:

1. Primary: human-anchored feasibility and soundness do not fall below C1.
2. Primary or key secondary after calibration: quality-qualified semantic
   yield exceeds C1 across the frozen threshold selected by blinded humans.
3. Controls: compare first to A0 and C1, then to C2/D1/D2 to isolate KL
   direction and trajectory source.
4. Replication: at least three training seeds with checkpoint seed as the unit.
5. Evaluation: matched prompts/K/decoding, independent human subset, full
   threshold curves, corrected NoveltyBench transfer, and qualitative failures.
6. Domain robustness: after the main result is stable, add one non-biomedical
   prompt domain rather than claiming the PubMed slice is universal.
7. Release: exact prompts, target rules, configs, raw judge records, model
   revisions, adapter hashes, seed-level effects, null findings, and exclusions.

The strongest paper framing is the quality--breadth trade-off in distilling
open-ended scientific ideation, not “automatic novelty improvement.”

The user's July 2026 paper,
[*Measuring the Gap Between Human and LLM Research Ideas*](https://arxiv.org/abs/2607.01233),
can contribute a **secondary research-taste analysis**: opportunity-pattern
and method-paradigm labels, normalized entropy,
TVD, and JSD. It should not become the success metric or a novelty oracle. That
paper conditions human/model ideas on reconstructed proximal literature and
author-audits a taxonomy subset; TOMATO directly states a question/background
and often an inspiration, so the prompt can nearly determine the taxonomy
label. Any use here needs a local two-human label audit and a within-prompt
entropy check proving the label still depends on the generated response.

## Current software and artifact verification

- The complete local test suite passes: 333 passed and two Torch-dependent
  tests were explicitly skipped because the lightweight local test environment
  does not install Torch. GPU jobs import and exercise their pinned Torch
  environments separately.
- Repository-wide Ruff checks pass, and all nine modified/new Python files are
  already Ruff-formatted.
- `git diff --check` reports no whitespace errors.
- The four new on-policy final adapters passed metadata, ordered-example,
  checkpoint-byte-identity, safetensors-header, and cross-node staging checks
  before evaluation.
- Held-out generation and judge validators fail closed unless all 1,658 prompt
  shards are complete; the aggregate analyzer is held until every declared
  checkpoint evaluation and cross-node repatriation hash gate succeeds.
- The semantic-calibration analyzer rejects the untouched label templates and
  writes no result: blank labels and null confidence values cannot be mistaken
  for human judgments.

These checks establish software and artifact integrity. They do not substitute
for human validation of the scientific rubric or semantic-equivalence boundary.

## Canonical artifact index

- Full validity audit and research path:
  [`EVALUATION_VALIDITY_AUDIT_20260805.md`](EVALUATION_VALIDITY_AUDIT_20260805.md)
- Seed-17 corrected machine summary:
  [`audits/corrected-k4-seed17-v2-summary.json`](audits/corrected-k4-seed17-v2-summary.json)
- B2a random-1 extension:
  [`compact-k4-seed17-b2a-extension/README.md`](compact-k4-seed17-b2a-extension/README.md)
- Corrected NoveltyBench matrix:
  [`audits/noveltybench-corrected-seed17-v2/README.md`](audits/noveltybench-corrected-seed17-v2/README.md)
- DeepInfra independent-judge result:
  [`DEEPINFRA_INDEPENDENT_JUDGE_FINDINGS_20260805.md`](DEEPINFRA_INDEPENDENT_JUDGE_FINDINGS_20260805.md)
- DeepInfra immutable manifest:
  [`audits/deepinfra-calibration-packet-20260805-v5.manifest.json`](audits/deepinfra-calibration-packet-20260805-v5.manifest.json)
- Target and temporal audit artifacts:
  [`audits/README.md`](audits/README.md)
- Literature claims and applicability limits:
  [`LITERATURE_SOURCE_LOG_20260805.md`](LITERATURE_SOURCE_LOG_20260805.md)
- Exact training/evaluation submission graphs:
  [`audits/compact-4b-1k-three-seed-training-submission-20260805.json`](audits/compact-4b-1k-three-seed-training-submission-20260805.json)
  and
  [`audits/compact-k4-three-seed-evaluation-submission-20260805.json`](audits/compact-k4-three-seed-evaluation-submission-20260805.json)
