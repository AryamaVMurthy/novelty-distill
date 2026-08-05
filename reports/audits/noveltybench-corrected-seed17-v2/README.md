# Corrected seed-17 NoveltyBench matrix

This bundle is the replacement for the withdrawn repeated-seed NoveltyBench
matrix. It compares the clean compact baselines with A0 on the full 100-prompt
curated split, with ten independently sampled generations per prompt.

## Protocol

- Benchmark: pinned `yimingzhang/novelty-bench` checkout
  `6a35510e530f236fd1dbcd9df888f01937c8494a`.
- Population: all 100 curated prompts.
- Generation seeds: 17--26 independently requested and observed for every
  prompt.
- Sampling: temperature 1.0, top-p 1.0, maximum 512 tokens.
- Scoring: NoveltyBench `quality_model=small`; one score record for every
  prompt and no unscored samples.
- Pairing: all deltas use the prompt as the paired unit. Values after `±` in
  the delta columns are standard errors of the 100 prompt-level differences.
- Inspect version used to read the logs: `inspect-ai==0.3.233`.

## Results

| Method | Distinct@10 mean ± SE | Utility@10 mean ± SE | Paired Δ diversity vs A0 ± SE | Paired Δ utility vs A0 ± SE |
|---|---:|---:|---:|---:|
| A0 | 4.080 ± 0.255 | 4.066 ± 0.196 | — | — |
| B2a random-1 SeqKD | 5.320 ± 0.277 | 4.720 ± 0.216 | +1.240 ± 0.204 | +0.655 ± 0.168 |
| B2b best-1 SeqKD | 5.520 ± 0.278 | 4.956 ± 0.221 | +1.440 ± 0.220 | +0.890 ± 0.190 |
| C1 off-policy forward KL | **5.580 ± 0.293** | **5.152 ± 0.227** | **+1.500 ± 0.192** | **+1.086 ± 0.164** |
| C2 off-policy reverse KL | 4.600 ± 0.265 | 4.624 ± 0.204 | +0.520 ± 0.159 | +0.558 ± 0.142 |
| D1 on-policy forward KL | 5.500 ± 0.292 | 5.121 ± 0.234 | +1.420 ± 0.196 | +1.056 ± 0.165 |
| D2 on-policy reverse KL | 4.510 ± 0.273 | 4.507 ± 0.205 | +0.430 ± 0.144 | +0.441 ± 0.137 |

The paired prompt fractions also show that the top-line gains are not produced
only by a few extreme items. C1 improves diversity on 71% of prompts, ties on
21%, and worsens on 8%; it improves utility on 68%, ties on 7%, and worsens on
25%. Its median prompt-level changes are +1 diversity and +0.814 utility, and
its 10%-trimmed utility change is +1.014. These median and trimmed summaries
are post-hoc descriptive robustness checks, not a new confirmatory test.

## Matched family contrasts

The content-bound paired outputs are preserved separately in this directory.

| Contrast | Paired Δ diversity ± SE | Paired Δ utility ± SE | Interpretation |
|---|---:|---:|---|
| D1 − C1 | −0.080 ± 0.128 | −0.030 ± 0.116 | On-policy forward KL has no visible transfer advantage over off-policy forward KL. |
| D2 − C2 | −0.090 ± 0.132 | −0.117 ± 0.110 | On-policy reverse KL is likewise tied with, or slightly below, its off-policy control. |
| B2b − B2a | +0.200 ± 0.128 | +0.236 ± 0.116 | Best-of-eight target selection may help generic transfer modestly, but this is a seed-17 descriptive result. |

The forward-KL pair C1/D1 leads this matrix. Both reverse-KL variants are much
closer to A0, consistent with a more mode-seeking response distribution. The
on-policy versions do not improve on their off-policy counterparts. This
objective ordering is useful baseline evidence, but seed-29 and seed-43 TOMATO
replications are still required before treating it as stable across training
runs.

## Cross-benchmark meaning

NoveltyBench contains generic prompts such as choosing a framework, writing a
poem, naming a country, selecting a ZIP code, and producing varied short
stories. It tests functional response diversity and benchmark utility. It does
not retrieve prior scientific work or decide whether a proposed research idea
is scientifically novel.

That boundary matters here. B2a and B2b improve over A0 on NoveltyBench even
though both have much lower feasibility and soundness on the 1,658-prompt
TOMATO evaluation. A model can therefore diversify generic responses while
being a poor scientific-idea generator. The two evaluation families are
complementary rather than interchangeable.

The largest C1 gains occur on open-ended or random-choice prompts, while some
losses occur on tightly constrained one-answer prompts. This is expected for a
diversity-oriented benchmark and reinforces the same claim boundary.

## Integrity checks

Every summary is validated against its raw `.eval` SHA-256, model identity,
100-sample completeness, K=10 configuration, and both declared and observed
seed schedule. The jobs all exited `0:0`:

| Method | Slurm job | Runtime | Mean unique raw completions | Minimum | Prompts with any exact duplicate |
|---|---:|---:|---:|---:|---:|
| A0 | 19105 | 46m44s | 9.39 | 2 | 18 |
| B2a | 19202 | 39m07s | 9.46 | 4 | 17 |
| B2b | 19198 | 39m48s | 9.49 | 4 | 16 |
| C1 | 19199 | 51m50s | 9.69 | 3 | 11 |
| C2 | 19200 | 52m31s | 9.52 | 3 | 17 |
| D1 | 19201 | 54m48s | 9.71 | 3 | 8 |
| D2 | 19203 | 51m05s | 9.59 | 3 | 13 |

Exact duplicates are legitimate concentration on constrained prompts, not the
old RNG defect: every completion request records the independent seed schedule
17--26.

## Files

- `matrix-vs-A0.json`: canonical seven-model summaries and 100-prompt paired
  deltas against A0.
- `forward-kl-paired.json`, `reverse-kl-paired.json`, and
  `seqkd-paired.json`: the three matched family contrasts.
- `METHOD/summary.json`: fail-closed canonical summary for each trained method.
- `METHOD/*.eval`: raw Inspect log containing prompts, generations, score
  records, metadata, declared seeds, and observed request events.

The A0 summary and raw log remain in
[`../noveltybench-corrected-full-A0-k10-seed17-v2/`](../noveltybench-corrected-full-A0-k10-seed17-v2/).
All analysis was generated by
[`../../../scripts/analyze_corrected_noveltybench.py`](../../../scripts/analyze_corrected_noveltybench.py).
