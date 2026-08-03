# Current research status

Last authoritative snapshot: 2026-08-04 00:12 IST.

## Research question

Can a Qwen3-4B student learn from a Qwen3-14B teacher while retaining several viable scientific
idea modes, rather than converging on one high-scoring or memorized response pattern?

The promotion decision requires held-out evidence on quality, semantic breadth, and copying. Job
completion, training loss, GPU memory, smoke scores, and unit-test counts are pipeline evidence and
do not select a method.

## Scientific evidence currently available

| Evidence | Result | What it establishes | What it does not establish |
|---|---|---|---|
| Frozen 1k teacher target bank | Eight teacher samples contain 3.798 measured modes/prompt on average; random-4 contains 2.584 and diverse-4 contains 3.146 | The training bank has nondegenerate semantic multiplicity and the selectors create different target distributions | That any student retains those modes |
| Untouched A0 temporal control | 26,528 ideas; composite judge mean 0.933896; feasibility 4.205858/5 and soundness 4.591337/5 | The untrained 4B reference is strong and the rubric has most useful headroom in feasibility and soundness | That A0 beats or loses to any trained method |
| A1 temporal generation | All 1,658 prompts x 16 samples completed and passed the idempotent generation preflight | The complete teacher comparison population exists under the frozen sampler | Teacher quality, breadth, or a student comparison; scoring is still running |

There is no trained-method winner yet. No trained student has completed the common held-out
quality-plus-semantic analysis.

## Execution gates

| Gate | State |
|---|---|
| Canonical TOMATO 1k/5k/20k data and 1,658 temporal test | Complete and replay-validated |
| TOMATO-1k training matrix | 15/19 complete; D3, E2, and E3 running; E4 queued |
| A0 temporal generation and scoring | Complete |
| A1 temporal generation | Complete: 1,658/1,658 prompt shards |
| A1 temporal scoring | Running: first pass job 18201, with job 18202 as idempotent resume |
| A0/A1 semantic-mode analysis | Waiting for complete A1 scores |
| Trained-method temporal evaluation | Waiting for all 19 training artifacts and the control gate |
| Research-taste appendix | 135 valid A0 shards retained; repaired 12-hour resume queued |
| TOMATO-5k promotion stage | Waiting for the frozen 1k comparison |
| TOMATO-20k, three seeds, official suites, and 1.7B/8B replication | Waiting for 5k selection |

## Results excluded from scientific comparison

- UltraFeedback runs and one-step smokes prove only trainer deployment.
- The cancelled negative-loss E2 trajectory used an invalid optional upstream clipping path and is
  excluded; corrected E2/E3/E4 train from the official unclipped objective.
- The first compact-decoder research-taste shards are excluded because the annotation interface
  failed its construct-validity diagnostic.
- Training-loss differences among D1/D2/D3 or OPSD variants do not measure idea quality.
- NoveltyBench and HypoSpace smoke scores use too few examples for model comparison.

## Next decision-producing artifact

After A1 scoring and all four remaining training runs complete, every method receives the same
1,658 prompts and 16-sample decoding contract. The first decision table will report feasibility,
soundness, teacher-mode recall and precision, student-only modes, ClusterJSD,
quality-adjusted coverage, and nearest-training-target similarity. Only that paired held-out table
can promote methods to TOMATO-5k.
