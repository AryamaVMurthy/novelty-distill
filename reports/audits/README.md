# Frozen validity-audit artifacts

These files are immutable, content-bound evidence produced before corrected evaluation reruns.

The consolidated interpretation and ICLR research path are in
[`../EVALUATION_VALIDITY_AUDIT_20260805.md`](../EVALUATION_VALIDITY_AUDIT_20260805.md).

- `tomato-target-integrity-20260805.json`: corpus-level repeated-passage and target-length audit.
- `tomato-pubmed-temporal-audit-20260805.json`: split-level earliest-public-date strata.
- `pubmed-metadata-20260805.json`: frozen PubMed XML-derived metadata used by the temporal audit.
- `noveltybench-corrected-smoke-A0-k10-seed17-v2/`: first portable independent-seed
  NoveltyBench implementation smoke, retained as raw Inspect evidence and a canonical summary.
- `noveltybench-corrected-full-A0-k10-seed17-v2/`: full 100-prompt A0 protocol gate, including
  the canonical summary and raw Inspect log. It passed and released the trained baseline runs.
- `noveltybench-corrected-seed17-submission-20260805.json`: job identities and concurrency
  contract for the corrected A0 plus six-method seed-17 NoveltyBench matrix.
- `noveltybench-corrected-seed17-v2/`: completed corrected matrix for A0 plus the six clean
  trained seed-17 baselines. It contains raw Inspect logs, content-bound summaries,
  prompt-paired deltas, matched family contrasts, and the construct-validity interpretation.
- `deepinfra-calibration-packet-20260805.manifest.json`: immutable identity of the first
  independent-judge packet. It is retained only for provenance and was superseded before paid
  use; see `deepinfra-calibration-packet-20260805.SUPERSEDED.md`.
- `deepinfra-calibration-packet-20260805-v3.manifest.json`: the failed `json_schema`
  transport probe and its bounded paid-request accounting. It is superseded.
- `deepinfra-calibration-packet-20260805-v4.SUPERSEDED.md`: the intermediate
  `json_object` protocol and its one persisted accepted probe, retired after the
  provider was observed to intermittently fence JSON.
- `deepinfra-calibration-packet-20260805-v5.manifest.json`: completed 528-call
  different-family judge run: 60 unique paired slots across eight methods and
  exactly six hidden repeats per method, with packet/run/analysis/response
  hashes, exact usage and cost, and verified node01 archive. Interpretation is
  in `../DEEPINFRA_INDEPENDENT_JUDGE_FINDINGS_20260805.md`.
- `semantic-equivalence-calibration-20260805-v2.manifest.json`: redacted provenance for 256
  similarity-stratified clean teacher pairs plus 26 hidden repeats per human rater. The first
  packet was superseded before labeling after its within-bin unique-prompt bug was detected.
- `semantic-equivalence-rater-ui-20260805.md`: hashes and handling instructions for the two
  source-blinded local browser raters generated from that frozen packet.
- `compact-4b-1k-three-seed-training-submission-20260805.json`: exact Slurm dependency graph
  for the clean six-method 4B replication at seeds 17/29/43 and 1,000 exposures.
- `compact-k4-three-seed-evaluation-submission-20260805.json`: exact matched-K=4 generation,
  score-gate, corrected-evaluation, and checkpoint-seed-aware analysis graph for seeds 29/43.
- `compact-three-seed-scheduling-amendment-20260805.md`: content-bound record of
  the resource-only training requeue, automated cross-node staging/release
  gates, judge-cache completion, evaluator OOM diagnosis, lower-memory
  replacement arrays, and downstream dependency verification.

The corrected K=4 compact summary is generated from the full remote evaluation JSONs and will be
stored here as `corrected-k4-seed17-v2-summary.json`. It contains aggregates, the complete
threshold curves, source hashes, and Slurm job IDs without duplicating bulky per-prompt records.

The associated report records interpretation, limitations, source-data hashes, and which legacy
results are quarantined. Do not overwrite these files; create a dated replacement after any
protocol or source-data change.
