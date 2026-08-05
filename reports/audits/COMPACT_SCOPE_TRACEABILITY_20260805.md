# Compact baseline scope traceability

Date: 2026-08-05 (Asia/Kolkata)

## Decision

The active experiment is the focused baseline/distillation study described in
`reports/FINAL_BASELINE_AND_VALIDITY_REPORT_20260805.md`. It is not the older
24-entry novelty-program graph in `reports/RESEARCH_RUN_LEDGER.md`. The older
ledger remains an immutable account of work attempted through 2026-08-04, but
its planned scale-ups and exploratory treatments are not outstanding work for
the compact study.

The current computational scope is complete: A0 plus B2a, B2b, C1-best1,
C2-best1, D1, and D2 have the required held-out results; all six trained
methods have independent checkpoint seeds 17, 29, and 43. The only gate for
promoting semantic-threshold diagnostics to human-anchored claims would be the
two-human semantic-equivalence calibration. The user deferred that work; the
automatic compact baseline is complete with those metrics quarantined. No GPU
run is waiting on the deferred gate.

## Why the historical ledger looks larger

The original registry contains four controls, nineteen executable training
variants, and one fail-closed method. Before the validity audit, it also
planned 5k/20k scaling, an exposure-sensitivity study, a second model-size
pair, research-taste annotation, and exploratory losses. On 2026-08-05 the
target-integrity, temporal-split, benchmark-RNG, judge, and semantic-threshold
audits changed the evidence policy. The current study retained only the clean,
minimal controls needed to compare hard sequence KD, token-level KL direction,
and static versus on-policy trajectories.

This scope decision prevents two errors:

1. treating a staged or historical experiment as an unfinished requirement;
2. treating an old checkpoint as valid evidence merely because training ran.

## Registry disposition

| Group | Historical execution | Current disposition |
|---|---|---|
| A0 | Untouched Qwen3-4B control generated and evaluated | Retained primary control; one frozen K=4 generation realization |
| A1 | Qwen3-14B teacher bank and held-out reference generated and evaluated | Retained as a teacher reference, not a trained student or three-seed control |
| B2a, B2b | Seed-17 training/evaluation plus seeds 29/43 replication | Retained; clean hard-KD controls, computationally complete |
| C1-best1, C2-best1 | Seed-17 training/evaluation plus seeds 29/43 replication | Retained; clean static forward/reverse-KL controls, computationally complete |
| D1, D2 | Seed-17 training/evaluation plus seeds 29/43 replication | Retained; genuine on-policy forward/reverse-KL controls, computationally complete |
| B1, A3, C1-human, C2-human | Historical-target artifacts were produced | Quarantined because the released historical target contains cross-topic passage contamination |
| E1 | Never executable under the pinned OPSD implementation | Correctly fail-closed, not a missing run |
| E2, E3 | E2 produced invalid partial steps before the vocabulary-clipping defect was found; a corrected one-step E2 gate passed. No accepted compact result exists | Quarantined because privileged context depends on the contaminated historical target; no rerun is required for the compact study |
| B2c, B3, B4, C1-diverse4, C2-diverse4, C3 | Seed-17 training artifacts were produced and system-validated | Historical exploratory artifacts only; not in the corrected compact contrast family and not required to be evaluated now |
| D3, E4 | Original graph reached partial/staged execution but produced no accepted compact-study result | Superseded exploratory variants; do not resume without a new protocol and explicit research question |
| F1 diversity-aware reverse KL, F2 random-4 | Protocol/code preparation only; no accepted current result | Future-method ideas, not compact-baseline obligations |
| 4k exposure sensitivity | Planned to resolve multi-target row/exposure confounding | Superseded with the diverse-4 methods; unnecessary for the single-target compact core |
| 5k/20k scaling | Original zero-runtime chains were cancelled and replacement chains were staged | Superseded; no accepted scale result and no current requirement to run it |
| 1.7B/8B replication | Infrastructure smoke work and a future replication plan | Superseded; add one robustness axis only after a new method has a stable 4B effect |
| Research-taste taxonomy | Compact-label pass was invalid; replacement work remained a secondary calibration-gated analysis | Superseded as a baseline endpoint; it may be revived only as a separately motivated appendix |

Historical training completion is therefore not the same as current evidence
eligibility. The exact seed-17 runtimes and artifact properties remain in the
run ledger and historical handoff for reproducibility.

## Evaluation disposition

| Evaluation | Status | Use now |
|---|---|---|
| TOMATO issue-month holdout, K=4 | Complete for A0/A1 and seed-17 clean methods; complete for seeds 29/43 of all six trained methods | Main operational baseline evidence; call it an issue-month holdout, not strict temporal OOD |
| Qwen3-32B rubric judge | Complete | Feasibility and soundness are operational proxy outcomes; saturated axes are secondary |
| Qwen3 embedding threshold curves | Complete, including all-threshold three-seed diagnostics | Descriptive and quarantined; human boundary calibration was deferred |
| DeepInfra Llama-3.3-70B audit | Complete: 528/528 valid calls, including 48 repeats | Different-family sensitivity check; confirms hard-KD failure but is too compressed to be ground truth |
| Original NoveltyBench matrix | Executed but invalid because ten calls repeated one seed | Withdrawn; never cite |
| Corrected NoveltyBench | Complete for A0 and the six clean trained methods, 100 prompts x 10 independent seeds | Generic functional diversity/utility transfer only, not scientific novelty |
| HypoSpace | Historical seven-model matrix audited | Secondary domain-specific constraint-search diagnostic; parser failures confound several rows |
| Human semantic equivalence | Packet and UI complete; labels absent | Deferred by the user; optional only if semantic metrics are later promoted to human-anchored claims |
| Literature-grounded global novelty | Not run | Intentionally outside the compact baseline study; would require retrieval and expert review |

## Deferred optional human calibration

This workflow is preserved for a future claim-bearing semantic analysis, but
it is not active work for the automatic compact baseline. Until it is revived
and completed, no embedding threshold is a human-calibrated boundary.

Two independent, domain-aware human raters must complete these local,
source-blinded browser files without seeing the private key, similarities,
repeat identities, or each other's labels:

- `artifacts/semantic-equivalence-calibration-20260805-v2/rater-one.html`
- `artifacts/semantic-equivalence-calibration-20260805-v2/rater-two.html`

Each rater labels 282 pairs: 256 unique-prompt originals plus 26 hidden
reversed repeats. The exported JSONL files replace copies of the blank local
templates. Build the source-blinded adjudication UI only after both exports
are complete:

```bash
uv run python scripts/build_semantic_equivalence_adjudicator.py \
  --packet artifacts/semantic-equivalence-calibration-20260805-v2/semantic-equivalence-public.json \
  --private-key artifacts/semantic-equivalence-calibration-20260805-v2/semantic-equivalence-private-key.json \
  --rater-one artifacts/semantic-equivalence-calibration-20260805-v2/labels-rater-one.jsonl \
  --rater-two artifacts/semantic-equivalence-calibration-20260805-v2/labels-rater-two.jsonl \
  --output artifacts/semantic-equivalence-calibration-20260805-v2/adjudicator.html
```

That UI contains only original disagreements or `uncertain` labels and exports
`adjudication.jsonl`. It excludes hidden repeats, similarities, private source
metadata, and the two initial labels. Then run the fail-closed analyzer:

```bash
uv run python scripts/analyze_semantic_equivalence_calibration.py \
  --private-key artifacts/semantic-equivalence-calibration-20260805-v2/semantic-equivalence-private-key.json \
  --rater-one artifacts/semantic-equivalence-calibration-20260805-v2/labels-rater-one.jsonl \
  --rater-two artifacts/semantic-equivalence-calibration-20260805-v2/labels-rater-two.jsonl \
  --adjudication artifacts/semantic-equivalence-calibration-20260805-v2/adjudication.jsonl \
  --output artifacts/semantic-equivalence-calibration-20260805-v2/analysis.json
```

The adjudication argument is omitted only when the two label files contain no
disagreement or uncertainty. Passing the gate requires Cohen's kappa at least
0.60 and adjudication of every flagged original. If kappa fails, the analyzer
writes an audit with `status: failed_inter_rater_gate`, records the diagnostic
`candidate_threshold`, leaves `selected_threshold` null, and exits nonzero.
The selected 0.80--0.99 embedding threshold calibrates pairwise semantic
equivalence only. It does not turn the metric into a measure of global
scientific novelty.

## Completion rule

The automatic compact baseline is computationally and analytically complete;
do not start additional GPU training under the old graph. If human calibration
is explicitly revived, then after the two exports and adjudication are
available, run the frozen analyzer, rerun semantic summaries at the selected
boundary, update the canonical report, and freeze those outputs with content
hashes. Any new loss, scale, dataset, or model size is a new study and requires
a new preregistered plan rather than an implicit continuation of the superseded
ledger.
