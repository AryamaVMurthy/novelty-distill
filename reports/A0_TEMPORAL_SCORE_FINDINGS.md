# A0 temporal score findings

This is a descriptive diagnostic for the untouched Qwen3-4B control, not a student comparison or
novelty result. CPU-only Slurm job 18600 validated and summarized all 1,658 score shards produced
by judge jobs 18203--18204. The corpus contains 26,528 generations (16 per temporal-test prompt)
and has score-tree SHA-256 `836367e24a98e4f52eb960573cbe6aa8eb8090d2236dd73c948dc3afacd5a82e`.

## Composite quality proxy

| Mean | Median | Population SD | Min | Max | Ceiling rate | Mean within-prompt range |
|---:|---:|---:|---:|---:|---:|---:|
| 0.933896 | 0.95 | 0.065451 | 0 | 1 | 0.253920 | 0.130157 |

The fixed Qwen3-32B-FP8 rubric is a reproducible proxy, not expert scientific judgment. These
values must not be compared directly with the training-target summary because the prompt sets
differ. Later model contrasts use the same temporal prompts and paired bootstrap analysis.

## Rubric discrimination

| Dimension | Mean | Rating-5 rate |
|---|---:|---:|
| Feasibility | 4.205858 | 0.254147 |
| Soundness | 4.591337 | 0.641172 |
| Relevance | 4.924043 | 0.926342 |
| Clarity | 4.957931 | 0.958798 |
| Instruction compliance | 4.998756 | 0.999246 |

Compliance, clarity, and relevance are near their ceilings. Feasibility retains the most rubric
headroom, followed by soundness. Consequently, the final report must preserve dimension-level
effects and cannot interpret a small composite difference without showing which axes changed.

## Generation diagnostics

- 26,522 generations ended naturally and six reached the 512-token bound, a length-stop rate of
  0.000226.
- Mean, median, and maximum completion lengths are 329.043, 328, and 512 tokens.
- Global and prompt-centered quality--length correlations are 0.0520 and 0.0701, respectively.

The negligible truncation rate removes length-cap saturation as a material A0 temporal confound.
Semantic-mode coverage, TasteGap diagnostics, and paired A0--A1/student contrasts remain pending.
