# Random-4 teacher-target diagnostic

This diagnostic was frozen and run before any `F2-random4` student metric was available. It
characterizes the targets selected from the canonical eight-sample TOMATO-1k teacher bank; it is
not evidence about trained-model performance.

## Provenance

- Producer commit: `033ca707`
- Dataset revision: `fcd201d92758a642465a7653b8055f3a04d5f439`
- Prompts: the frozen 1,000-prompt TOMATO-1k set
- Random selector: deterministic, uniform without replacement, nested so that random-4 contains
  B2a's exact random-1 target
- Result artifact:
  `/scratch/aryama.murthy/novelty-distill/evaluations/teacher-targets-random4-teacher-1k-v1/analysis.json`
- Semantic modes: the preregistered automatic teacher-target clusters, not human labels

## Results

| Target view | Mean quality | Modes / prompt | Prompts with >=4 modes | Length-stop rate | Mean completion tokens |
|---|---:|---:|---:|---:|---:|
| Random-1 | 0.827300 | 1.000 | 0.000 | 0.001000 | 356.216 |
| Random-4 | 0.826950 | 2.584 | 0.213 | 0.001750 | 355.604 |
| Diverse-4 | 0.866088 | 3.146 | 0.542 | 0.003000 | 357.571 |
| All-8 | 0.827656 | 3.798 | 0.542 | 0.001875 | 355.549 |

Random-4 versus random-1 changes mean quality by only -0.000350 and mean completion length by
-0.612 tokens, while adding 1.584 modes per prompt. It is therefore a clean value-independent
multiplicity target for the matched-exposure student contrast.

Diverse-4 versus random-4 adds 0.039138 mean quality, 0.562 modes per prompt, and 32.9 percentage
points of prompts with at least four modes. Consequently, a future B3 advantage over the repeated
single-target baseline would not by itself identify a response-multiplicity effect: B3 changes
both response count and selection. The declared `F2-random4` versus `B2a-4x` edge isolates
multiplicity, while B3-4x versus F2 isolates the additional coverage/quality selection effect.

All-8 has essentially the same mean quality as random-4 (+0.000706) but 1.214 more modes per
prompt. This supports the interpretation that the quality increase in diverse-4 comes from its
selector rather than merely from drawing more teacher responses. It does not establish that the
student will retain those modes; that question remains gated on identical temporal generation,
scoring, clustering-threshold sensitivity, TasteGap-style distributional analysis, and later
multi-seed confirmation.
