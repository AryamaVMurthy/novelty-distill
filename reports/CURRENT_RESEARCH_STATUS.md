# Current research status

> **Superseded on 2026-08-05 by a post-hoc validity audit.** Do not cite the
> conclusions below as current evidence. The old NoveltyBench matrix used a
> repeated generation seed and is invalid; `B1` and all historical-human-target
> methods are quarantined because TOMATO target text contains cross-topic
> passage contamination; the issue-month split is not a strict earliest-public
> temporal split; and the reported `viable_semantic_yield` omitted feasibility
> from its gate. Feasibility and soundness scores for prompt-only methods remain
> descriptive frozen-Qwen-judge measurements. See
> [`audits/README.md`](audits/README.md) and the dated validity report before
> using any number in this file.

## Current corrected snapshot

Jobs 19055--19062 completed the feasibility-inclusive K=4 recomputation for
A0, A1, B1, B2b, C1, C2, D1, and D2. The content-bound summary is
[`audits/corrected-k4-seed17-v2-summary.json`](audits/corrected-k4-seed17-v2-summary.json),
and the complete current interpretation is
[`EVALUATION_VALIDITY_AUDIT_20260805.md`](EVALUATION_VALIDITY_AUDIT_20260805.md).
The usable pilot reference is C1; B2b is the hard sequence-KD failure control,
C2 is the static reverse-KL control, and D1/D2 are on-policy controls. B1 is
descriptive only and quarantined. The missing B2a random-1 SeqKD matched-K=4
extension is complete after all four score shards and the global 1,658-prompt
gate passed. B2a random-1 and B2b best-1 SeqKD are effectively tied at the low
quality level (feasibility 3.398; soundness 3.165--3.176), showing that hard
single-target SeqKD—not best-of-eight selection—is the seed-17 failure mode.
The complete handoff is
[`compact-k4-seed17-b2a-extension/README.md`](compact-k4-seed17-b2a-extension/README.md).
The current different-family DeepInfra v3
packet is ready: 60 unique paired prompts across eight methods, 480 originals,
and 48 hidden repeats balanced at six per method. Paid judging awaits a rotated
key exported through the environment. Its immutable identity is
[`audits/deepinfra-calibration-packet-20260805-v3.manifest.json`](audits/deepinfra-calibration-packet-20260805-v3.manifest.json).
The separate semantic-equivalence v2 packet is also prepared from 28,000 clean
teacher-pair candidates: 256 unique-prompt originals, 32 in each of eight
similarity strata, and 26 hidden repeats per rater. It awaits two blinded human
raters and is audited in
[`audits/semantic-equivalence-calibration-20260805-v2.manifest.json`](audits/semantic-equivalence-calibration-20260805-v2.manifest.json).

The corrected full A0 NoveltyBench gate also passed: Distinct@10 4.080 and
Utility@10 4.066 over 100 prompts, with the full seed 17--26 schedule observed.
The exact duplicates on 18 prompts are concentrated in constrained factual
questions and are not the old repeated-seed bug. The six clean trained
seed-17 methods are now running on the same corrected protocol. See
[`audits/noveltybench-corrected-full-A0-k10-seed17-v2/README.md`](audits/noveltybench-corrected-full-A0-k10-seed17-v2/README.md).

## Historical 2026-08-04 snapshot

The compact K=4 baseline study, the
D2 on-policy reverse-KL extension, and the separate seven-model
NoveltyBench/HypoSpace transfer matrix are complete. External transfer results
are reported separately from the TOMATO results below.

## TL;DR

The basic baseline ladder is now finished end to end on 1,658 held-out TOMATO
prompts. The best trained baseline to carry forward is off-policy forward KL
(`C1-best1`). Hard sequence KD (`B2b`) collapses viable breadth, reverse KL
(`C2-best1`) preserves quality but is more mode-seeking, and on-policy forward
KL (`D1`) does not improve over `C1-best1`. On-policy reverse KL (`D2`) also
preserves high quality but is the narrowest KL baseline: its viable semantic
yield is 0.486 below D1 and 0.511 below C1-best1.

This is a baseline result, not a novelty claim. The metrics are operational
frozen-judge and embedding outcomes, not human validation of scientific
novelty.

The final external aggregate is [`official-matrix-seed17.md`](official-matrix-seed17.md).
It uses one declared seed, so its means are descriptive and its population SD is
zero by construction; it is not a multi-seed significance analysis.

## What was compared

| ID | Role |
|---|---|
| `A0` | untouched Qwen3-4B control |
| `A1` | frozen Qwen3-14B teacher anchor used to define reference modes |
| `B1` | human-supervised SFT |
| `B2b` | hard sequence KD, best-of-8 teacher target |
| `C1-best1` | off-policy forward-KL distillation |
| `C2-best1` | off-policy reverse-KL distillation |
| `D1` | on-policy forward-KL distillation |
| `D2` | on-policy reverse-KL distillation |

Every student row uses the same 1,658 held-out prompts and K=4 sampling budget.
The evaluation judge is frozen `Qwen/Qwen3-32B-FP8`; it is separate from the
Qwen3-14B training teacher.

## Completed levels

| Method | Feasibility /5 | Soundness /5 | Teacher recall | Viable yield /4 |
|---|---:|---:|---:|---:|
| `A0` | 4.218 | 4.603 | 0.120 | 1.900 |
| `B1` | 3.636 | 3.603 | 0.025 | 1.799 |
| `B2b` | 3.398 | 3.176 | 0.031 | 0.565 |
| `C1-best1` | 4.239 | 4.657 | 0.134 | 2.248 |
| `C2-best1` | 4.240 | 4.656 | 0.135 | 1.795 |
| `D1` | 4.234 | 4.658 | 0.117 | 2.223 |
| `D2` | 4.231 | 4.638 | 0.125 | 1.737 |

## Official transfer matrix

Each model generated 1,000 NoveltyBench answers and 610/90/350 HypoSpace causal/3D/Boolean
answers under the pinned seed-17 protocol. NoveltyBench `distinct_k_mean` is 1.0 for every model,
so utility is the informative NoveltyBench number. The compact summary below reports
NB utility followed by HypoSpace causal validity and Boolean validity; the complete metric table
is in [`official-matrix-seed17.md`](official-matrix-seed17.md).

| Method | NB utility | Causal validity | 3D validity | Boolean validity |
|---|---:|---:|---:|---:|
| `A0` | 1.841757 | 0.222951 | 0.111111 | 0.594286 |
| `B1` | 1.590812 | 0.118033 | 0.055556 | 0.597143 |
| `B2b` | 1.772299 | 0.227869 | 0.111111 | 0.631429 |
| `C1-best1` | 1.819351 | 0.283607 | 0.000000 | 0.648571 |
| `C2-best1` | 1.855201 | 0.270492 | 0.100000 | 0.602857 |
| `D1` | 1.908975 | 0.126230 | 0.000000 | 0.605714 |
| `D2` | 1.870885 | 0.254098 | 0.022222 | 0.634286 |

This transfer matrix has no universal winner: D1 has the highest NoveltyBench utility but weak
causal/3D validity; C1-best1 has the strongest causal and Boolean validity but zero 3D validity;
and D2 has strong causal/Boolean transfer with low 3D validity. These are benchmark-specific
descriptive outcomes, not novelty claims.

## Predeclared paired conclusions

- `B1 - A0`: worse on all four primary outcomes.
- `B2b - B1`: worse feasibility, soundness, and viable yield; recall change is
  not distinguishable from zero.
- `C1-best1 - B2b`: large improvements on all four primary outcomes.
- `C2-best1 - C1-best1`: feasibility, soundness, and recall are tied; viable
  yield is lower by 0.453, 95% CI [-0.496, -0.410].
- `D1 - C1-best1`: no feasibility, soundness, or viable-yield improvement;
  recall is lower by 0.0164, 95% CI [-0.0295, -0.0031].

The separately predeclared D2 extension finds:

- `D2 - D1`: feasibility and recall are not distinguishable; soundness is
  lower by 0.0204 and viable yield is lower by 0.486, 95% CI
  [-0.528, -0.441].
- `D2 - C2-best1`: feasibility and recall are not distinguishable; soundness
  is lower by 0.0178 and viable yield is lower by 0.0579, 95% CI
  [-0.0971, -0.0181].
- `D2 - C1-best1`: feasibility and recall are not distinguishable; soundness
  is lower by 0.0195 and viable yield is lower by 0.511, 95% CI
  [-0.554, -0.466].

Holm-corrected p-values and all threshold curves are in
[`compact-k4-seed17/findings.md`](compact-k4-seed17/findings.md).
The D2 extension's corrected tests and threshold curves are in
[`compact-k4-seed17-d2-extension/findings.md`](compact-k4-seed17-d2-extension/findings.md).

## Compact-study audit status

- All 20 generation tasks and all 20 scoring tasks exited 0.
- All five generation gates and all five score gates validated 1,658 complete
  prompts with zero pending.
- All five student evaluations and the paired analysis exited 0.
- D2 added 6,632 complete generations and judge records; its generation and
  score gates, embedding evaluation, and three-contrast analysis all exited 0.
- Compact CSV/SVG artifact hashes match their manifest.
- The compact-study jobs and the separate external transfer matrix are complete; no
  benchmark job remains in the queue.

The canonical compact handoff is
[`compact-k4-seed17/README.md`](compact-k4-seed17/README.md); full job and
provenance details are in
[`compact-k4-seed17/RUN_AUDIT.md`](compact-k4-seed17/RUN_AUDIT.md).
The complete D2 handoff and recovery audit are in
[`compact-k4-seed17-d2-extension/README.md`](compact-k4-seed17-d2-extension/README.md)
and
[`compact-k4-seed17-d2-extension/RUN_AUDIT.md`](compact-k4-seed17-d2-extension/RUN_AUDIT.md).

## Next research decision

Do not add another broad matrix yet. Use `C1-best1` as the main distillation
baseline, `C2-best1` as the off-policy mode-seeking control, and `D1`/`D2` as the
on-policy KL-direction controls. D2 is the strongest observed mode-seeking
control. The next method should state one explicit hypothesis for improving
viable semantic yield without reducing feasibility or soundness, then compare
directly against `C1-best1` under this same frozen evaluation.
