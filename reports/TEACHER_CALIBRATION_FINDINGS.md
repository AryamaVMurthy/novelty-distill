# Teacher calibration findings

These results are calibration evidence, not trained-student or headline paper results. The frozen
design and promotion rules are in `reports/TEACHER_CALIBRATION_PROTOCOL.md`.

## Runs and provenance

| Study | Generation | Judge | Embedding/analysis | Result artifact |
|---|---:|---:|---:|---|
| Seven-condition decoding matrix | 17946 | 17950 | 17951; CPU analysis 17966 | `evaluations/teacher-calibration/qwen3-14b-v1/analysis.json` |
| Paired concision intervention | 17968 | 17970 | 17972 | `evaluations/teacher-calibration/qwen3-14b-concision-v1/analysis.json` |
| Concise temperature follow-up | 18000 | 18012 | 18014 | `evaluations/teacher-calibration/qwen3-14b-concise-temperature-v1/analysis.json` |

All paths are below `/scratch/aryama.murthy/novelty-distill` on Turing node01. Both studies used
the same eight frozen prompt IDs, eight independently seeded outputs per prompt, pinned
Qwen3-14B generation, Qwen3-32B-FP8 judging, Qwen3-Embedding-4B embeddings, and deterministic
SGLang inference. The first cluster job produced its seven score/cluster artifacts but did not
persist its final report; the report was regenerated without model calls from those immutable
artifacts in CPU job 17966. The cluster launcher now creates and verifies both final report files.

## Decoding matrix

| Condition | Mean quality | Clusters at 0.82 | Quality-adjusted coverage | Length stop |
|---|---:|---:|---:|---:|
| temperature 0.6, top-p 0.8, 512 | 0.954 | 1.125 | 1.081 | 100.0% |
| **temperature 0.7, top-p 0.8, 512** | **0.954** | **1.250** | **1.212** | **100.0%** |
| temperature 0.8, top-p 0.8, 512 | 0.956 | 1.375 | 1.325 | 100.0% |
| temperature 1.0, top-p 0.8, 512 | 0.956 | 1.125 | 1.081 | 100.0% |
| temperature 1.2, top-p 0.8, 512 | 0.958 | 1.375 | 1.325 | 100.0% |
| temperature 0.8, top-p 0.95, 512 | 0.959 | 1.250 | 1.206 | 100.0% |
| temperature 0.7, top-p 0.8, 768 | 0.967 | 1.000 | 0.988 | 95.3% |

No condition met the predeclared promotion rule. The apparent cluster gain at temperatures 0.8
and 1.2 was only +0.125 mode per prompt, did not occur on a majority of prompts, and was not stable
across the threshold curve. Raising the cap from 512 to 768 tokens reduced termination by only
4.7 percentage points and lowered coverage. The legacy top-p 0.95 setting added no primary-threshold
modes. Therefore the official Qwen non-thinking controls (temperature 0.7, top-p 0.8, top-k 20,
min-p 0) remain the decoding reference.

Clustering is strongly threshold-sensitive. At the instructed-embedding threshold 0.82, the
reference averaged 1.25 modes; at 0.90 it averaged 3.125, and at 0.95 it averaged 5.125. The fixed
0.82 result is evidence of mode collapse under that declared operational definition, not evidence
that 0.82 is a universally valid semantic boundary.

## Paired concision intervention

The follow-up changed only the effective response requirement: one hypothesis and test plan, at
most 300 words, with an explicit mechanism, distinction, intervention, measurable outcomes, and
falsifier. The paired control repeated official 0.7/0.8 decoding without that requirement. This
study used the subsequently anchored judge rubric for both arms, so its quality scores must not be
compared numerically with the first study.

| Outcome | Unconstrained | Concise | Paired difference | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Length termination | 1.000 | 0.000 | -1.000 | [-1.000, -1.000] |
| Completion tokens | 512.0 | 299.4 | -212.6 | not tested |
| Mean quality | 0.946 | 0.973 | +0.027 | [+0.016, +0.036] |
| Clusters at 0.82 | 1.250 | 1.000 | -0.250 | [-0.750, 0.000] |
| Quality-adjusted coverage | 1.206 | 1.000 | -0.206 | [-0.681, +0.044] |

All 64 concise outputs ended normally, all obeyed the 300-word limit, and their mean length was
about 203 words. The quality gain came primarily from feasibility (+0.547 on the 1--5 dimension
scale); soundness was unchanged. Seven prompts had unchanged primary cluster counts, while one
prompt fell from three clusters to one. Concision therefore fixes truncation and improves judged
testability, but does not by itself satisfy the semantic-coverage promotion rule.

## Concise temperature follow-up

The final calibration held the successful concise response instruction fixed and changed only
temperature. All conditions used top-p 0.8, top-k 20, min-p 0, a 512-token cap, and the same
eight prompts and eight seeds.

| Temperature | Mean quality | Clusters at 0.82 | Quality-adjusted coverage | Length stop |
|---:|---:|---:|---:|---:|
| **0.7** | **0.973** | **1.000** | **1.000** | **0.0%** |
| 0.8 | 0.964 | 1.000 | 1.000 | 0.0% |
| 1.0 | 0.970 | 1.000 | 0.994 | 0.0% |
| 1.2 | 0.975 | 1.000 | 1.000 | 0.0% |

No prompt gained a primary-threshold cluster under any alternative temperature. Temperature 0.8
had a paired quality difference of -0.0094 versus 0.7 (95% bootstrap CI [-0.0156, -0.0039]);
its raw paired-test p-value was 0.0356 but Holm-adjusted p-value was 0.1068. Temperatures 1.0 and
1.2 had neither a reliable quality advantage nor any primary-threshold coverage gain. None met the
predeclared promotion rule.

## Decision and next gate

Production teacher data therefore uses the concise official temperature 0.7/top-p 0.8 setting.
This choice is supported by termination and judged-quality evidence, not by a demonstrated
diversity increase. The `diverse4` target remains a sensitivity baseline selected from all eight
teacher samples; it must not be described as a proven four-mode treatment. All eight samples are
retained so future clustering or expert annotation can revisit the operational diversity boundary.

## Validity limits

- Eight length-stratified prompts are a calibration sample, not a population estimate.
- The fixed judge and embedding model are automatic proxies; no expert semantic labels exist yet.
- Connected-component clustering can chain samples and is especially sensitive near the cosine
  boundary.
- The anchored rubric improved within-prompt judge dispersion, but controlled corruptions or human
  ratings are still needed to establish evaluator calibration.
- The 1k/5k/20k training and untouched 1,658-record temporal test artifacts are validated
  separately; none of these calibration scores should enter final model-comparison tables.

## Verified source log

- Qwen3 official quickstart (software documentation): non-thinking decoding recommends temperature
  0.7, top-p 0.8, top-k 20, min-p 0, and `enable_thinking=False`.
  <https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md>
- SGLang official server arguments (software documentation): deterministic inference enables
  batch-invariant operations.
  <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>
- Qwen3-Embedding-4B official model card (model documentation): clustering is supported and an
  English task instruction is recommended.
  <https://huggingface.co/Qwen/Qwen3-Embedding-4B>
