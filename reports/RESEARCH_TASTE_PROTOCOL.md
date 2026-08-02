# Research-taste secondary analysis protocol

## Status and source

This protocol was added on 2026-08-02 after training and teacher-target generation had begun but
before any temporal-test student metric was available. It is therefore a **secondary descriptive
analysis** and cannot replace, alter, or promote findings from the frozen primary outcomes.

- Citation key: `chen2026tastegap`
- Ziyu Chen, Yilun Zhao, and Arman Cohan, *Measuring the Gap Between Human and LLM Research
  Ideas*, arXiv:2607.01233, 2026.
- Status: preprint; arXiv v1 dated 2026-07-01.
- Canonical paper: <https://arxiv.org/abs/2607.01233>
- Author repository: <https://github.com/ziyuuc/TasteGap>
- Relevance: introduces a two-axis taxonomy that distinguishes semantic variety from variety in
  how research opportunities are framed and contributions are constructed.
- Limitation for reuse: the public author repository had no explicit software license when checked
  on 2026-08-02. This repository independently implements the evaluation contract and does not
  copy the author implementation.

Chen et al. report that their evaluated LLMs concentrate on bridge-like opportunities and
synthesis/unification methods relative to realized human-paper ideas. Their task, model set,
annotator, and literature inputs differ from TOMATO, so those percentages are motivation for this
analysis rather than expected values or evidence about our outputs.

## Research question

Does distillation preserve or change two different kinds of scientific-idea breadth?

1. **Semantic breadth:** distinct mechanisms and experimental tests, measured by the frozen
   complete-linkage semantic-mode protocol.
2. **Research-taste breadth:** distinct problem-framing and contribution strategies, measured by
   the attributed Chen--Zhao--Cohan taxonomy.

A method can cover several semantic modes while repeatedly using one bridge-and-synthesis template.
Conversely, it can use several research paradigms while remaining semantically close to one
mechanism. The two measurements must therefore be reported separately.

## Frozen taxonomy

Opportunity patterns describe why work is needed:

1. puzzle or contradiction;
2. explanation gap;
3. assumption or scope mismatch;
4. measurement or evidence gap;
5. fragmentation or bridge opportunity;
6. failure or risk gap;
7. resource or operational constraint.

Method paradigms describe how the contribution is constructed:

1. explicit synthesis or unification;
2. assumption relaxation or scope extension;
3. failure mitigation or robustification;
4. formal or conceptual derivation;
5. measurement or empirical mapping;
6. constructive artifact or system;
7. optimization, search, or resource strategy.

The annotator additionally records surface stitching, bottleneck specificity, and boilerplate on
the same ordinal scale defined in `configs/evaluation/research_taste.yaml` and the strict schema in
`src/novelty_distill/evaluation/research_taste.py`.

## Comparisons and outputs

Every A3 historical human response, A1 teacher sample, A0 control sample, and executable trained
method sample is annotated. For each method, report:

- global category shares and normalized entropy on both axes;
- total-variation distance and base-2 Jensen--Shannon divergence from A3 human taste;
- the same distances from A1 teacher taste;
- bridge-opportunity and synthesis-method rates;
- mean surface-stitching, bottleneck-specificity, and boilerplate diagnostics;
- mean opportunity, method, and joint category coverage among K samples per prompt;
- the change in human JSD relative to A1, where a negative value means closer to A3 than A1;
- both the complete K-sample view and deterministic sample-index-zero one-shot sensitivity view.

Uncertainty uses 10,000 prompt-level paired bootstrap resamples at seed 17. Each sampled prompt
contributes its within-prompt category proportions, so K completions are not treated as independent
units. Intervals are pointwise 95% percentile intervals and this secondary analysis does not add a
new confirmatory p-value family.

All comparisons require exactly the same temporal prompt population. K samples are repeated draws
within a prompt and are never treated as independent research problems.

## Pre-result expectations

- A1 may already differ materially from A3, so matching the teacher is not automatically human-like
  research taste.
- Diverse-4 training may improve semantic coverage without improving taste entropy; disagreement is
  scientifically informative rather than a failed metric.
- Human-target B1 and human-privileged E2 may be closer to A3 than teacher-only methods, but this is
  exploratory because the protocol was added after training began.
- If on-policy methods sharpen common student templates, they may have lower taste entropy even when
  their quality or semantic precision improves.

No direction becomes a confirmatory claim in the current 1k gate.

## Reliability gate

The paper validated GPT-5.4-mini labels against two humans; that agreement does not transfer to our
pinned Qwen3-32B-FP8 annotator. Before any headline research-taste claim, at least 150 records must
be independently labeled by at least two human annotators. The automated labeler must achieve
Cohen's kappa of at least 0.80 on both taxonomy axes. Calibration records must be stratified across
A3, A1, A0, and trained sources without revealing method identities to annotators.
Every pair of human annotators must meet the same threshold; a high model--human score cannot pass
the gate when the human labeling contract itself is unreliable. The deterministic 150-record
packet, hidden source key, and analysis tooling are produced by
`scripts/prepare_research_taste_calibration.py` and
`scripts/analyze_research_taste_calibration.py`.

If the gate fails, automatic taste results remain explicitly descriptive appendix material. The
frozen primary quality and semantic-mode analyses remain valid and unchanged.

## Threats to validity

- TOMATO provides a research background/question and sometimes inspirations, whereas Chen et al.
  reconstruct titles and abstracts of proximal prior work. Absolute distributions are not directly
  comparable across the two datasets.
- A3 contains one realized historical response per prompt while generative methods contain K=16.
  The sample-zero view controls sampling count; the all-sample view measures behavioral breadth.
- Taxonomy labels compress mixed research moves into one primary category per axis.
- LLM annotation errors may correlate with fluency, model family, or terminology despite blinding.
- Human taste is a reference distribution, not a monotonic definition of scientific value. A method
  should not be optimized to imitate historical frequencies without quality and expert review.

## Future treatment, not part of the frozen gate

A later `TasteKD` ablation may select teacher targets jointly by judge quality, semantic-mode
coverage, and research-taste coverage, or generate candidates conditioned on underrepresented
taxonomy cells. That treatment must be declared and trained only after the present evaluation; it
cannot be retroactively inserted as a primary baseline.
