# Frozen validity-audit artifacts

These files are immutable, content-bound evidence produced before corrected evaluation reruns.

The consolidated interpretation and ICLR research path are in
[`../EVALUATION_VALIDITY_AUDIT_20260805.md`](../EVALUATION_VALIDITY_AUDIT_20260805.md).

- `tomato-target-integrity-20260805.json`: corpus-level repeated-passage and target-length audit.
- `tomato-pubmed-temporal-audit-20260805.json`: split-level earliest-public-date strata.
- `pubmed-metadata-20260805.json`: frozen PubMed XML-derived metadata used by the temporal audit.
- `noveltybench-corrected-smoke-A0-k10-seed17-v2/`: first portable independent-seed
  NoveltyBench implementation smoke, retained as raw Inspect evidence and a canonical summary.

The corrected K=4 compact summary is generated from the full remote evaluation JSONs and will be
stored here as `corrected-k4-seed17-v2-summary.json`. It contains aggregates, the complete
threshold curves, source hashes, and Slurm job IDs without duplicating bulky per-prompt records.

The associated report records interpretation, limitations, source-data hashes, and which legacy
results are quarantined. Do not overwrite these files; create a dated replacement after any
protocol or source-data change.
