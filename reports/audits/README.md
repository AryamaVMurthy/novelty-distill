# Frozen validity-audit artifacts

These files are immutable, content-bound evidence produced before corrected evaluation reruns.

- `tomato-target-integrity-20260805.json`: corpus-level repeated-passage and target-length audit.
- `tomato-pubmed-temporal-audit-20260805.json`: split-level earliest-public-date strata.
- `pubmed-metadata-20260805.json`: frozen PubMed XML-derived metadata used by the temporal audit.

The associated report records interpretation, limitations, source-data hashes, and which legacy
results are quarantined. Do not overwrite these files; create a dated replacement after any
protocol or source-data change.
