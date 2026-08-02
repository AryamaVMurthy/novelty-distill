# Production teacher-target findings

This report freezes the teacher-only semantic-mode calibration and derived training views before
any TOMATO-1k student model is evaluated. Embedding clusters are operational proxies, not expert
labels of scientific novelty or mechanism equivalence.

## Artifact integrity

- Cluster job 18127 rebuilt the production artifact from the frozen score tree using 16,000 cached
  raw/instructed embeddings and completed in 00:01:45. Validation jobs 18128 and 18129 completed in
  00:00:02 and 00:00:03.
- The artifact reconstructs exactly for 1,000 prompt IDs and 8,000 generations at seed 17. The
  prompt file SHA-256 is `995c9648a9b591fbae381a2f3f2de265f25a85ebb3e7ba891309403692ae9ccf`.
- The clustered JSONL SHA-256 is
  `716bb28c8b3663492b401412309255cad9c74809595e3de076fd5c9f45b546df`; the target artifact
  SHA-256 is `0ac24f018c54fd8998b5fb4f39e06584c9af4b26ec075983e07736f615c563f4`.
- Target production used Git commit `efe65337ce9e92749303e76bfab7838e42d1de35`; the final analysis
  and validation used `6ab33c6ec9da828ec55751ae89765e25e936b18e`.

## Pre-student threshold recalibration

The original cosine threshold 0.82 yielded only 1.001 instructed clusters per prompt, with 999 of
1,000 prompts collapsed to one teacher mode. That would make ModeRecall and the intended
diverse-four target contrast nearly degenerate. Before any student output was generated or
inspected, the primary threshold was moved to 0.95, an endpoint already present in the declared
0.70--0.95 sensitivity grid.

At 0.95, instructed clusters have mean 3.558 and median 3 per prompt. The distribution is:

| Modes | Prompts |
|---:|---:|
| 1 | 279 |
| 2 | 140 |
| 3 | 128 |
| 4 | 100 |
| 5 | 114 |
| 6 | 90 |
| 7 | 87 |
| 8 | 62 |

This makes the primary analysis nondegenerate, but 0.95 is not a universal semantic boundary. At
that threshold the raw embedding has 4.001 modes/prompt versus 3.558 for the instructed embedding;
the instruction increases, preserves, and lowers the raw count on 168, 415, and 417 prompts. Every
claim must therefore retain the full threshold curve and seek expert confirmation.

## Derived-view trade-offs

| View | Responses | Quality mean | Length-stop rate | Modes/prompt | At least four modes |
|---|---:|---:|---:|---:|---:|
| `random1` | 1,000 | 0.827300 | 0.001000 | 1.000 | 0.000 |
| `best1` | 1,000 | 0.916600 | 0.006000 | 1.000 | 0.000 |
| `mode1` | 1,000 | 0.898600 | 0.006000 | 1.000 | 0.000 |
| `diverse4` | 4,000 | 0.868625 | 0.003000 | 2.755 | 0.453 |
| `all8` | 8,000 | 0.827656 | 0.001875 | 3.558 | 0.453 |

Best-1 raises the judge proxy by 0.0893 over deterministic random-1. Mode-1 retains a 0.0713 gain
over random-1 while differing from best-1 on 284 prompts. Diverse-4 trades some selected-response
quality for broader mode coverage, but only 45.3% of prompts contain at least four measured modes;
when fewer exist, the view deterministically fills the remaining slots with high-quality responses.
The label `diverse4` therefore describes the four-response selection policy, not a guarantee of
four expert-distinct scientific mechanisms.

The immutable machine-readable artifacts are on Turing under
`evaluations/teacher-target-validation-teacher-1k-v1.json` and
`evaluations/teacher-targets-teacher-1k-v1/analysis.json`.
