# Current research status

Last authoritative snapshot: 2026-08-04. The compact K=4 baseline study and
the D2 on-policy reverse-KL extension are complete; no cluster jobs remain.

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

## Audit status

- All 20 generation tasks and all 20 scoring tasks exited 0.
- All five generation gates and all five score gates validated 1,658 complete
  prompts with zero pending.
- All five student evaluations and the paired analysis exited 0.
- D2 added 6,632 complete generations and judge records; its generation and
  score gates, embedding evaluation, and three-contrast analysis all exited 0.
- Compact CSV/SVG artifact hashes match their manifest.
- The Turing queue is empty.

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
