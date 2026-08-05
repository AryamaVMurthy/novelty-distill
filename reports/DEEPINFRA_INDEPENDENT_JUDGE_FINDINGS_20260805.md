# DeepInfra independent-judge sensitivity analysis

Date: 2026-08-05 (Asia/Kolkata)

## Bottom line

The different-family Llama judge independently reproduces one important
direction from the frozen Qwen judge: both hard single-target sequence-KD
students (`B2a`, `B2b`) have much worse **soundness** than A0, the teacher, and
all four KL baselines. It also reproduces the tie between random-1 and best-1
sequence KD and finds no meaningful separation among `C1`, `C2`, `D1`, and
`D2` on soundness.

It does **not** validate the absolute scores. The Llama judge is more generous
and substantially more compressed than Qwen: 99.8% of clarity scores are 5,
95.6% of relevance scores are 5, 92.9% of feasibility scores are exactly 4,
76.0% of soundness scores are 5, and it marks zero fatal flaws. Feasibility
rank agreement with Qwen is weak (Spearman 0.231); soundness agreement is only
moderate (0.583). The outcome is therefore useful as a robustness check for
the large hard-KD soundness failure, not as ground truth, human validation, or
a scientific-novelty measure.

## Frozen design

- Provider/model: DeepInfra,
  `meta-llama/Llama-3.3-70B-Instruct-Turbo`.
- Source: frozen seed-17 corrected K=4 generations from A0, A1, B2a, B2b,
  C1-best1, C2-best1, D1, and D2.
- Paired sample: the same 60 unique prompt/sample slots for every method,
  selected across pooled Qwen feasibility/soundness rank quartiles.
- Original records: 60 per method, 480 total.
- Blinded repeats: 6 per method, 48 total.
- Calls: 528, temperature 0, one request at a time, one allowed attempt.
- Judge input: the complete scientific task plus one candidate answer. Method,
  checkpoint, Qwen scores, repeat identity, and other private metadata were not
  sent.
- Judge rubric: relevance, feasibility, soundness, clarity, instruction
  compliance, fatal flaw, and a brief rationale. The prompt explicitly says
  that literature novelty cannot be verified and must not be claimed.

The immutable packet, run, analysis, response fingerprint, provider usage, and
node01 archive are recorded in
[`audits/deepinfra-calibration-packet-20260805-v5.manifest.json`](audits/deepinfra-calibration-packet-20260805-v5.manifest.json).

## Method scores

Each row is the mean of the 60 paired original slots. Qwen values are the
private frozen scores used to form the sample; Llama values are the independent
responses.

| Method | Qwen feasibility | Llama feasibility | Qwen soundness | Llama soundness | Llama fatal flaws |
|---|---:|---:|---:|---:|---:|
| A0 | 4.200 | 4.083 | 4.667 | 4.917 | 0/60 |
| A1 | 4.700 | 4.067 | 4.933 | 4.933 | 0/60 |
| B2a | 3.383 | 3.983 | 3.200 | 4.233 | 0/60 |
| B2b | 3.417 | 4.033 | 3.183 | 4.217 | 0/60 |
| C1-best1 | 4.217 | 4.083 | 4.617 | 4.917 | 0/60 |
| C2-best1 | 4.167 | 4.083 | 4.617 | 4.950 | 0/60 |
| D1 | 4.133 | 4.133 | 4.733 | 4.917 | 0/60 |
| D2 | 4.183 | 4.033 | 4.617 | 4.950 | 0/60 |

The key pattern is visible in soundness: Llama raises the hard-KD scores by
about a full point but still leaves them roughly 0.7 below the KL methods. In
feasibility, compression is strong enough that the teacher is no higher than
A0 and the hard-KD penalty shrinks from roughly 0.8--1.3 Qwen points to only
0.05--0.15 Llama points. Absolute Llama feasibility should not be used to
declare the hard-KD outputs feasible.

## Paired contrasts

The analysis resamples the 60 paired prompt/sample-slot differences 5,000
times. The intervals below describe this frozen sensitivity sample; they are
not a substitute for training-seed replication or human assessment.

| Contrast (right minus left) | Feasibility mean [95%] | Soundness mean [95%] | Interpretation |
|---|---:|---:|---|
| B2a minus A0 | -0.100 [-0.183, -0.033] | -0.683 [-0.817, -0.550] | hard random-1 KD is worse |
| B2b minus A0 | -0.050 [-0.133, 0.017] | -0.700 [-0.817, -0.583] | hard best-1 KD is worse in soundness |
| B2b minus B2a | 0.050 [-0.017, 0.117] | -0.017 [-0.117, 0.083] | random/best hard KD remain tied |
| C1 minus B2a | 0.100 [0.017, 0.183] | 0.683 [0.533, 0.817] | forward KL repairs the hard-KD failure |
| C1 minus B2b | 0.050 [-0.033, 0.133] | 0.700 [0.567, 0.833] | same conclusion for best-1 KD |
| D1 minus C1 | 0.050 [-0.017, 0.133] | 0.000 [-0.067, 0.067] | no on-policy forward-KL advantage |
| C2 minus C1 | 0.000 [-0.083, 0.083] | 0.033 [-0.033, 0.100] | KL directions tied on these scores |
| D2 minus C2 | -0.050 [-0.117, 0.017] | 0.000 [-0.067, 0.067] | no on-policy reverse-KL advantage |
| D2 minus D1 | -0.100 [-0.183, -0.033] | 0.033 [-0.050, 0.133] | small lower D2 feasibility; soundness tied |

This supports carrying C1 as the main quality-preserving distillation baseline
and B2a/B2b as hard-KD failure controls. It does not establish that C1 is more
novel or semantically more diverse; the independent judge never evaluates
those constructs.

## Judge agreement and saturation

| Dimension | Qwen mean | Llama mean | Exact agreement | Within one | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| relevance | 4.717 | 4.956 | 0.748 | 0.992 | 0.260 | 0.313 | 0.264 |
| feasibility | 4.050 | 4.063 | 0.596 | 0.992 | 0.413 | 0.236 | 0.231 |
| soundness | 4.321 | 4.754 | 0.571 | 0.929 | 0.500 | 0.613 | 0.583 |
| clarity | 4.785 | 4.998 | 0.796 | 0.992 | 0.213 | 0.083 | 0.089 |
| instruction compliance | 4.983 | 4.975 | 0.958 | 1.000 | 0.042 | -0.021 | -0.021 |

The high within-one rates are partly mechanical on a compressed five-point
scale and must not be read as strong validation. Soundness contains the most
shared ordering signal. Feasibility contains little. Clarity and compliance
are effectively saturated for both judges.

Llama's original-record histograms are:

| Dimension | score 1 | score 2 | score 3 | score 4 | score 5 | SD | ceiling rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| relevance | 0 | 0 | 0 | 21 | 459 | 0.205 | 95.6% |
| feasibility | 0 | 0 | 2 | 446 | 32 | 0.259 | 6.7% |
| soundness | 0 | 0 | 3 | 112 | 365 | 0.445 | 76.0% |
| clarity | 0 | 0 | 0 | 1 | 479 | 0.046 | 99.8% |
| instruction compliance | 0 | 0 | 0 | 12 | 468 | 0.156 | 97.5% |

The fatal-flaw rate is 0/480. That field and the clarity/compliance dimensions
are already fully or nearly fully fulfilled and provide almost no useful model
separation. Relevance is also too compressed. Soundness remains informative
only for a large failure such as hard KD. Feasibility requires human review.

## Reliability and rationale behavior

Hidden-repeat exact agreement is 100% for relevance, clarity, and compliance,
95.8% for feasibility, and 97.9% for soundness; every repeat is within one
point. This shows high self-consistency at temperature 0. It does not show
correctness.

Only 218 of 480 rationales are exact-unique, for a 54.6% exact-duplicate rate.
They average 6.7 words and never exceed the 40-word contract. The judge often
uses generic phrases such as “clear hypothesis” or “feasible test plan.” This
is additional evidence that the rationales are lightweight format checks, not
deep domain-expert reviews.

### Qualitative disagreement audit

High-disagreement cases show what the compression hides. These examples were
located after the complete blinded run and are diagnostic, not a new scoring
rule:

- A B2a answer about Paleolithic lead exposure proposes only “administer lead
  exposure to a controlled population,” without defining a population,
  ethically credible intervention, evolutionary identification strategy, or a
  way for a present experiment to establish historical selection. Qwen assigns
  feasibility/soundness 2/2; Llama assigns 4/4 and says “sound mechanism, and
  feasible test plan.”
- A B2b answer to an observational ICA-occlusion comparison proposes
  randomizing acute patients to standard care versus an unspecified
  intervention such as thrombolysis. That intervention does not identify the
  requested proximal-versus-nonproximal comparison. Qwen assigns 2/2; Llama
  assigns 4/4 and describes only “minor gaps.”
- A B2a WNT7 answer claims a DVL-independent receptor mechanism but tests only
  GPR124 knockout, which cannot by itself distinguish the claimed direct-FZD
  mechanism from alternatives. Qwen assigns 2/2; Llama assigns 4/4 despite
  noting that biochemical evidence is absent.

All three receive `fatal_flaw=false`. This is concrete evidence that the Llama
judge rewards the presence of hypothesis/mechanism/test-plan headings and may
underweight whether the intervention actually identifies the claimed
mechanism. It explains why the independent feasibility scale cannot replace
domain-aware human review.

## Transport, completion, and cost

- All 528 request IDs are unique.
- Finish reason is `stop` for 528/528; length-stop rate is 0.
- All 528 raw provider responses reparse to exactly the saved scores and model.
- The model returned raw JSON only 21 times and a single whole-response JSON
  fence 507 times. Protocol v3 deliberately accepts only those two forms and
  then applies strict schema validation; preambles, extra prose, multiple
  fences, extra keys, and wrong types remain rejected.
- Usage is present for 528/528 responses: 499,146 prompt tokens, 38,070
  completion tokens, 537,216 total tokens.
- DeepInfra's exact summed estimated cost for the accepted v5 run is USD
  0.062097. Earlier superseded transport probes are accounted separately and
  are not mixed into this figure.

## What this does and does not resolve

Resolved:

- The large seed-17 hard-KD soundness collapse is not merely a same-family
  Qwen-judge artifact; a different model family sees the same direction.
- Random-1 versus best-1 hard KD remains a tie, so best-of-eight target
  selection is not the main explanation for that failure.
- No quality advantage for current on-policy forward or reverse KL appears
  under either automatic judge.
- The earlier concern about judge saturation is confirmed, not removed.

Unresolved:

- Neither model can determine global scientific novelty without retrieval and
  expert comparison to prior work.
- Two LLM judges agreeing does not establish human alignment.
- The semantic-equivalence boundary used for diversity/yield is still awaiting
  two blinded domain-aware human raters.
- These 60 slots come from one seed-17 checkpoint per method. The running
  seeds 29 and 43 are required for across-training-seed uncertainty.

Recent work specifically warns that reference-free judges can be overly
generous, that scientific-novelty judgments can oppose domain experts, and
that inter-LLM agreement or geometric consistency does not imply human
alignment. The verified sources and their applicability limits are logged in
[`LITERATURE_SOURCE_LOG_20260805.md`](LITERATURE_SOURCE_LOG_20260805.md).
