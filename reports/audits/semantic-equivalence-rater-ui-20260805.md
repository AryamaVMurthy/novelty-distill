# Semantic-equivalence rater UI — 2026-08-05

Two self-contained, source-blinded browser files were generated from the frozen
v2 public calibration packet. Each presents one pair at a time, stores progress
only in the browser's local storage, requires a label and 1--3 confidence value,
and exports analyzer-compatible JSONL only after every item is complete.

| Rater | Local ignored file | SHA-256 |
|---|---|---|
| one | `artifacts/semantic-equivalence-calibration-20260805-v2/rater-one.html` | `030384e0dfc50f957ab84b87f07d2f3c172039856f3f788cdc3f4ca4bf64d870` |
| two | `artifacts/semantic-equivalence-calibration-20260805-v2/rater-two.html` | `1e97916c3bb69c5c7511a6cdd8685059b6ac97817aa39fc80f09a46f481c1763` |

Each file contains 282 judgments: 256 unique-prompt originals and 26 hidden
repeats. The two raters must work independently and must not receive the private
key, similarity values, bins, source identities, or each other's labels.

The committed rater generator is
`scripts/build_semantic_equivalence_rater.py`. Generated HTML remains ignored
because it embeds the full public annotation packet. The exported files should
replace the empty local templates `labels-rater-one.jsonl` and
`labels-rater-two.jsonl` only after a copy has been preserved.

After both exports exist,
`scripts/build_semantic_equivalence_adjudicator.py` validates exact coverage,
uses the private key only to remove hidden repeats, and creates a second
self-contained browser UI containing only original disagreements or
uncertainties. It does not embed similarities, bins, prompt IDs, source
indices, repeat identities, or either rater's labels. The adjudicator exports
`adjudication.jsonl`. If there are no flagged originals, run the analyzer
without `--adjudication`.

The final analyzer now records a candidate threshold for audit but returns a
nonzero exit and leaves `selected_threshold` null when Cohen's kappa is below
0.60. A boundary is selected only after the preregistered agreement gate
passes.
