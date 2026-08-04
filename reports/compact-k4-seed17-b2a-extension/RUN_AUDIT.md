# B2a corrected K=4 run audit

## Execution graph

| Stage | Slurm jobs | Status |
|---|---|---|
| Generation | 19087, 19094, 19095, 19093 | all completed, exit 0 |
| Global generation gate | 19096 | completed, exit 0 |
| Four-shard Qwen judging | 19114, 19115, 19116, 19097 | all completed, exit 0 |
| Global score gate | 19098 | completed, exit 0 |
| Corrected embedding evaluation | 19099 | completed in 5m26s, exit 0 |
| Predeclared paired analysis | 19100 | completed in 6s, exit 0 |

The first generation array suffered three shared-environment failures; its one
successful shard was retained and the other three shards were recovered in a
fresh array. The global gate subsequently validated the whole run. This is why
the four completed generation task IDs are not from one contiguous array.

## Completeness and identity

- Held-out prompts: 1,658.
- Generations: 6,632 (K=4), with deterministic prompt/sample coordinates.
- Judge records: 6,632 from frozen `Qwen/Qwen3-32B-FP8`.
- Generation and score files: exactly 1,658 each after the global gates.
- Student checkpoint: `checkpoints/B2a-tomato1k-seed17/final`, bound at
  generation time to artifact identity
  `sha256:812cb0cbb2b8bc6efbd5163aeb5c54b2c74b0cd4f400ba8b703cd7a62b2aa7f5`.
- Corrected evaluation SHA-256:
  `89842fefc0c57547034c69b3c8c87fe9e5c4b7bd7a3ee665cb1ea462e4b137fd`.
- Analysis producer commit:
  `68d35683732075fa1ccb55f2b687026c055a3821`.

No length truncation was observed (`length_stop_rate=0`). Semantic metrics
retain the frozen 0.70--0.95 curve, but single-threshold inference is disabled
until the blinded human equivalence calibration is completed.

## B2a versus B2b diagnostic

The 1,000 training targets are not aliases: 11% of random-1 targets equal the
best-1 target, near the 12.5% chance rate for eight candidates. Random indices
occur 111--144 times each; best-1 indices are concentrated earlier (50--345),
consistent with actual score selection rather than a copied view.

Across 6,632 paired held-out completion slots, B2a/B2b exact text equality is
0.000905. Their mean Qwen deltas (B2b minus B2a) are nevertheless only +0.00075
for feasibility and +0.01071 for soundness; mean completion length differs by
-0.33 tokens. Thus their matched aggregate behavior is not caused by the same
output artifact.
