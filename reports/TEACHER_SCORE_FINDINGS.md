# Production teacher-score findings

This report records the frozen Qwen3-32B-FP8 rubric scores for the 1,000-prompt teacher run. The
scores are reproducible proxy measurements of response quality; they are not novelty labels or
expert judgments of scientific value.

## Artifact integrity

- Score jobs 18035 and 18037 completed successfully in 01:29:07 and 00:00:49 respectively. The
  second job performed a strict preflight and exited without loading the judge.
- The score tree contains exactly 1,000 schema-v2 shards and 8,000 records, with eight contiguous
  sample indices per prompt, one pinned judge specification, and 8,000 unique request IDs.
- Strict loading verified every response-text SHA-256 binding and the arithmetic mean of all five
  rubric dimensions. The complete score tree SHA-256 is
  `3c2986fe0a6ca2a985737a177f2e6fc3ed3ac3e669c553f0dfa7a5b437989e7f`.
- The frozen judge is `Qwen/Qwen3-32B-FP8` revision
  `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`, with thinking disabled, temperature zero, strict
  structured output, and 128 maximum judge tokens.

## Score behavior

Across 8,000 responses, the five-dimension quality mean is 0.827656, the median is 0.85, and the
population standard deviation is 0.104174. Scores span 0.45--1.00; 5.85% reach the aggregate
ceiling. The mean within-prompt score range is 0.19215 and the mean number of distinct scores among
eight responses is 3.269, so the aggregate judge is not globally collapsed.

| Dimension | Mean | Rating-5 rate | Interpretation |
|---|---:|---:|---|
| Relevance | 4.472875 | 0.498875 | High, with material headroom |
| Feasibility | 3.733625 | 0.060125 | Strongest useful separation from the ceiling |
| Soundness | 3.714250 | 0.128750 | Strong useful separation from the ceiling |
| Clarity | 4.640000 | 0.644125 | Partially ceiling-limited |
| Instruction compliance | 4.992375 | 0.992500 | Saturated; unsuitable as a discriminating headline |

Instruction compliance therefore serves mainly as a format-integrity check. Feasibility and
soundness retain considerably more discrimination and must remain visible beside the aggregate
quality score.

## Length diagnostics

There are 7,985 normal stops and 15 length stops, a truncation rate of 0.001875. Completion length
has mean 355.549 tokens, median 353, and maximum 512. The global quality--length Pearson correlation
is 0.078351 and the prompt-centered correlation is 0.111392. These weak positive associations do
not indicate that length dominates the judge score, but they are nonzero and remain a declared
confound in method comparisons.

The immutable machine-readable analysis is stored on Turing under
`evaluations/score-diagnostics-teacher-1k-v1/analysis.json`; it was produced by Git commit
`de758795ff9112c63c4c35708f765d47c8273ccd`.
