# HypoSpace component validity audit

Date: 2026-08-05 (Asia/Kolkata)

## Decision

The accepted HypoSpace artifacts can be retained as **secondary,
task-specific constraint-satisfaction transfer diagnostics**. They must not be
called scientific novelty measurements, combined into a universal score, or
used as primary ICLR evidence. Their strongest signal is often exact-format
compliance rather than hypothesis-search ability.

This decision is independent of the withdrawn NoveltyBench component. The
HypoSpace requests were sequential and adaptive: every query saw the parsed
previous hypotheses and was asked to avoid repetition. They did not use the
NoveltyBench adapter's erroneous repeated-call construction.

## Artifact and routing checks

- The benchmark repository is pinned at
  `c69e9318577b34b5b896996571aefd4ba6053f58`.
- Every accepted method used the same 61 causal, 9 3D, and 35 Boolean sample
  identifiers, with no duplicate identifier within a domain.
- Every sample received ten sequential queries and every accepted artifact
  reports seed `33550336` and zero provider errors.
- A0 records `OpenRouter(novelty-model)`. Every adapter run records
  `OpenRouter(novelty-base:novelty-model)`, the SGLang syntax required to route
  to the loaded LoRA. The frozen model manifests bind the methods to distinct
  artifact hashes.
- The accepted 3D runs use the pinned revision plus the recorded NumPy
  compatibility patch. This is not an untouched-upstream result.
- The 21 accepted raw artifacts were copied from node01/node02/node03 for this
  audit. The SHA-256 of the sorted `sha256sum` listing is
  `382cf7688c2d3a074a22072da5ebfcf3c2e6772a404b266cadecde6805274e18`.

## What the metrics mean

- `parse_success`: fraction of requests that produced the exact domain schema.
- `validity`: fraction of all ten requests satisfying the observed constraints;
  parse failures therefore count as invalid.
- `uniqueness`: unique in-space structures divided by ten. This is structural
  diversity within a synthetic hypothesis space, not scientific novelty.
- `recovery`: macro-average fraction of enumerated compatible ground truths
  recovered per task.

The official protocol is an adaptive search procedure, not ten independent
draws: parsed hypotheses are inserted into the next prompt. That is a coherent
HypoSpace design, but no per-request backend seed is recorded, so exact
run-to-run sampling reproducibility and variance remain unmeasured.

## Parse confounding

The table reports unconditional validity and a diagnostic `valid | parsed`
rate computed from the raw per-sample counts. The latter is not a replacement
official metric; it exposes how much of an apparent failure is formatting.

| Method | Domain | Parse | Validity | Valid given parsed |
|---|---|---:|---:|---:|
| A0 | causal | 0.970 | 0.223 | 0.230 |
| B1* | causal | 0.143 | 0.118 | 0.828 |
| B2b | causal | 1.000 | 0.228 | 0.228 |
| C1-best1 | causal | 0.836 | 0.284 | 0.339 |
| C2-best1 | causal | 1.000 | 0.270 | 0.270 |
| D1 | causal | 0.230 | 0.126 | 0.550 |
| D2 | causal | 1.000 | 0.254 | 0.254 |
| A0 | 3D | 0.889 | 0.111 | 0.125 |
| B1* | 3D | 0.800 | 0.056 | 0.069 |
| B2b | 3D | 1.000 | 0.111 | 0.111 |
| C1-best1 | 3D | 0.111 | 0.000 | 0.000 |
| C2-best1 | 3D | 0.333 | 0.100 | 0.300 |
| D1 | 3D | 0.011 | 0.000 | 0.000 |
| D2 | 3D | 0.222 | 0.022 | 0.100 |
| A0 | Boolean | 1.000 | 0.594 | 0.594 |
| B1* | Boolean | 1.000 | 0.597 | 0.597 |
| B2b | Boolean | 1.000 | 0.631 | 0.631 |
| C1-best1 | Boolean | 1.000 | 0.649 | 0.649 |
| C2-best1 | Boolean | 1.000 | 0.603 | 0.603 |
| D1 | Boolean | 1.000 | 0.606 | 0.606 |
| D2 | Boolean | 1.000 | 0.634 | 0.634 |

`B1` remains quarantined because its training target is contaminated. Its raw
transfer behavior is retained only to make the audit complete.

The causal and 3D artifacts contain 1,438 unparsed requests in total (1,111
causal and 327 3D). Upstream stores parsed hypotheses and provider exceptions,
but it does not store the raw text of a syntactically unparseable completion.
Consequently, the large B1/D1 causal and C1/D1 3D format failures cannot be
reclassified or qualitatively diagnosed from the accepted artifacts. This is
a material auditability limitation.

## Boolean summarizer correction

The pinned Boolean CLI records `parse_success_rate` per sample but omits the
top-level aggregate. The old local summarizer filled the missing field with
1.0 whenever provider errors were zero, which is logically invalid because a
successful HTTP request can still fail parsing. The summarizer now computes
the mean from explicit per-sample parse rates and fails closed if they are
missing or out of range. All seven accepted Boolean artifacts actually have
per-sample parse rate 1.0, so this code correction does not change their
reported numbers.

## Paper use

If HypoSpace is retained, report its three domains separately, put parse rate
beside validity, and describe it as synthetic structured-search transfer. Do
not average domains, do not interpret `uniqueness` as research novelty, and do
not spend the next compute budget replicating HypoSpace before the core
three-seed TOMATO quality/breadth study is sound. A future rerun should preserve
every raw completion and explicit request seed.
