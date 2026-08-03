# Four-target exposure sensitivity protocol

This secondary study is frozen before any TOMATO-1k student evaluation metric is available. It
does not modify the 19-method primary matrix or its Holm family. Its purpose is to separate access
to four distinct teacher targets from simply receiving four times as many optimizer-example
exposures.

## Matched matrix

Every run uses the primary Qwen3-4B initialization, seed 17, the same 1,000 ordered TOMATO prompt
IDs, teacher-target artifact, optimizer, learning rate, context limit, and non-thinking rendering.
All six runs receive exactly 4,000 optimizer-example exposures:

| ID | Target rows available | Exposure interpretation |
|---|---|---|
| `B2a-4x` | one random teacher target per prompt | four passes over the single-target rows |
| `B2b-4x` | one best teacher target per prompt | four passes over the single-target rows |
| `B2c-4x` | one mode teacher target per prompt | four passes over the single-target rows |
| `B3-4x` | four diverse teacher targets per prompt | one full pass over all four-target rows |
| `B4-4x` | four diverse teacher targets per prompt | one full GEM pass over all four-target rows |
| `F2-random4` | four uniformly selected teacher samples from the canonical eight | one full pass over four random-target rows |

The corresponding primary runs received 1,000 exposures. Training rows and exposures remain
separate provenance fields; no row is relabeled or duplicated in the saved dataset.

`F2-random4` was added on 2026-08-03, before any trained-model temporal metric was available,
after Amin et al.'s *Escaping the Mode Lottery: Multi-Response Training Improves Language Model
Generalization* (arXiv:2606.00544v1) identified Random-K-of-N as the unbiased reference selector
for distributional fine-tuning. The four indices are selected uniformly without replacement by a
deterministic prompt-ID/seed hash; neither response text, judge quality, nor semantic clusters enter
the selector. The random permutation is nested: its first retained sample is exactly B2a's frozen
random-1 target, and the other three are drawn from the remaining seven. This retains a uniform
Random-4 subset while reducing selector noise in the multiplicity contrast. The author's public implementation had no explicit software license at commit
`8f389da41ad3c8441c9569f746a9a040cf388ddc` when checked on 2026-08-03, so this repository uses
an independent, dependency-free implementation rather than copying its code.

## Evaluation and interpretation

The models use the identical temporal K=16 generation, judge, embedding, complete-linkage, and
threshold-sensitivity contracts as the primary matrix. Declared secondary contrasts are:

- `B3-4x` against each of `B2a-4x`, `B2b-4x`, and `B2c-4x`;
- `F2-random4` against repeated `B2a-4x` to isolate response multiplicity under value-independent
  selection;
- `B3-4x` against `F2-random4` to isolate coverage-aware versus unbiased random-4 selection;
- `B4-4x` against `B3-4x`;
- each 4x run against its exact 1x counterpart.

A diverse-target result is exposure-robust only when its direction is not explained away by the
matched repeated-single-target controls and remains stable across the clustering threshold curve.
The study is still seed-17 screening evidence; it cannot substitute for the later multi-seed final
matrix or expert validation of scientific novelty.

The jobs are low-priority, bounded, resumable, and begin only after the frozen primary controller.
They cannot delay completion of primary training or enter the primary contrast family.

Primary source: <https://arxiv.org/abs/2606.00544>
