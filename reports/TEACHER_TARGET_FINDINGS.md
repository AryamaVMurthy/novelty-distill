# Production teacher-target findings

This report freezes the teacher-only semantic-mode calibration and derived training views before
any TOMATO-1k student model is evaluated. Embedding clusters are operational proxies, not expert
labels of scientific novelty or mechanism equivalence.

## Artifact integrity

- Final cluster job 18131 rebuilt the artifact from the frozen score tree using 16,000 cached
  raw/instructed embeddings and completed in 00:01:52. Final validation gate 18132 completed in
  00:00:03.
- The artifact reconstructs exactly for 1,000 prompt IDs and 8,000 generations at seed 17. The
  prompt file SHA-256 is `995c9648a9b591fbae381a2f3f2de265f25a85ebb3e7ba891309403692ae9ccf`.
- The clustered JSONL SHA-256 is
  `45e88c3e2c44e7e59d7a54e554bb3ccbe9eeea6dfbba8a5fe802a3ed52b43d73`; the target artifact
  SHA-256 is `eea6b6e35f5d5e181740f656a5ec49a29ed40daa21ba3297ab26dc0ffc2fbd9f`.
- Target production, analysis, and validation all used Git commit
  `6bc49ef5c857b2e00d797996681709fd6654ffaa`.
- The primary partition is deterministic complete linkage at instructed-embedding cosine 0.94.
  Every pair of samples within a measured teacher mode therefore meets the similarity threshold;
  a bridge sample cannot chain two otherwise dissimilar modes together.

## Pre-student linkage and threshold audit

The original connected-component threshold 0.82 yielded only 1.001 instructed clusters per
prompt, with 999 of 1,000 prompts collapsed to one teacher mode. Moving connected components to
0.95 yielded 3.558 modes, but deterministic qualitative inspection found clear paraphrases split
apart and exposed the sharp bridge-chaining transition between 0.90 and 0.95.

Before any student output was generated or inspected, complete and average linkage were evaluated
on the frozen 8,000 teacher embeddings. Complete linkage was chosen because it directly enforces
the pairwise similarity boundary and prevents chaining. At 0.94 it yields mean 3.798 and median 4
instructed modes per prompt, while deterministic extreme-case inspection grouped obvious duplicate
fMRI-connectivity and nanoparticle-delivery hypotheses that connected components at 0.95 split.

The final primary-mode distribution is:

| Modes | Prompts |
|---:|---:|
| 1 | 114 |
| 2 | 168 |
| 3 | 176 |
| 4 | 200 |
| 5 | 137 |
| 6 | 116 |
| 7 | 73 |
| 8 | 16 |

This makes the primary analysis nondegenerate, but 0.94 is not a universal semantic boundary. At
that threshold the raw embedding has 4.232 modes/prompt versus 3.798 for the instructed embedding;
the instruction increases, preserves, and lowers the raw count on 145, 390, and 465 prompts. Every
claim must therefore retain the complete eight-threshold curve and seek expert confirmation.

## Derived-view trade-offs

| View | Responses | Quality mean | Length-stop rate | Modes/prompt | At least four modes |
|---|---:|---:|---:|---:|---:|
| `random1` | 1,000 | 0.827300 | 0.001000 | 1.000 | 0.000 |
| `best1` | 1,000 | 0.916600 | 0.006000 | 1.000 | 0.000 |
| `mode1` | 1,000 | 0.893450 | 0.004000 | 1.000 | 0.000 |
| `diverse4` | 4,000 | 0.866088 | 0.003000 | 3.146 | 0.542 |
| `all8` | 8,000 | 0.827656 | 0.001875 | 3.798 | 0.542 |

Best-1 raises the judge proxy by 0.0893 over deterministic random-1. Mode-1 retains a 0.06615 gain
over random-1 while differing from best-1 on 397 prompts. Diverse-4 trades some selected-response
quality for broader mode coverage, but only 54.2% of prompts contain at least four measured modes;
when fewer exist, the view deterministically fills remaining slots with high-quality responses.
The label `diverse4` therefore describes the four-response selection policy, not a guarantee of
four expert-distinct scientific mechanisms.

The immutable machine-readable artifacts are on Turing under
`evaluations/teacher-target-validation-teacher-1k-v1.json` and
`evaluations/teacher-targets-teacher-1k-v1/analysis.json`.
