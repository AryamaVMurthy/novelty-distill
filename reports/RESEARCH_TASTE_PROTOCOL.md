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
- Limitation for reuse: the public author repository at commit
  `c5da2aac43006b723b2fb0d0c8e92e2d42a4501e` had no explicit software license when rechecked on
  2026-08-03. The public IdeaSeed dataset snapshot
  `2d44e6eded453f50cc98a2790e0e891fd9a0f7cf` also declared no license in its Hugging Face card.
  This repository therefore independently implements the evaluation contract, does not copy the
  author implementation, and does not download or redistribute IdeaSeed in the automatic run
  graph. An external IdeaSeed transfer benchmark remains conditional on explicit reuse terms.

Chen et al. report that their evaluated LLMs concentrate on bridge-like opportunities and
synthesis/unification methods relative to realized human-paper ideas. Their task, model set,
annotator, and literature inputs differ from TOMATO, so those percentages are motivation for this
analysis rather than expected values or evidence about our outputs.

## Annotation-interface amendment

The first A0 annotation pass was stopped and invalidated before any research-taste comparison was
run. Its compact regular-expression decoder assigned 12,805/14,928 outputs (85.78%) to scope
mismatch and gave an
obviously mechanism-specific stroke example a specificity score of 1/3. On a deterministic,
label-stratified 20-record diagnostic sample, the compact decoder agreed with the same pinned model
using an exact-JSON but unconstrained decoder on only 45% of opportunity labels; mean bottleneck
specificity was 1.05 versus 2.75. The unconstrained labels are not treated as ground truth, but the
large same-model interface sensitivity disproves the compact decoder as a stable measurement.

Jobs 18415/18416 were therefore cancelled, and all 933 completed prompt shards were preserved at
`research-taste-invalid-compact-v1/A0-temporal-k16-seed17000` on scratch. They are excluded from
every analysis input. The replacement protocol removes forced label-token decoding, incorporates
the paper's axis-separation and decision guidance, asks for the exact six-key JSON object, and
strictly validates either raw JSON or one enclosing JSON Markdown fence. The changed protocol hash
prevents old shards from passing the replacement preflight. This repair occurred before any
trained-model temporal output or research-taste outcome was analyzed.

The first replacement live smoke (job 18583) then exposed one narrower contract defect: without a
literal key example, the model renamed `boilerplate_score` and omitted two required diagnostic
fields. Strict validation terminated the job before its first shard, and its resume job 18584 was
cancelled after 12 seconds. The protocol now includes all six literal key names; a regression test
requires every schema key to appear in the request. These zero-shard smoke failures are deployment
evidence only and never enter the result matrix.

Replacement job 18585 (commit `d8c89f3`, 12-hour limit) then wrote the first validated shard under
protocol hash `55d9ba2b...`: 16 records, 16 unique request IDs, the exact schema, and no retry.
Idempotent resume 18586 is staged after any first-pass exit. Early per-prompt labels are monitored
only for deployment anomalies; annotation validity and any cross-method claim still require the
blinded human gate.

The replacement still uses Qwen3-32B-FP8 rather than the paper's GPT-5.4-mini annotator and TOMATO
does not provide the paper's separated prior-work/motivation/method input contract. Those transfer
differences remain subject to the human reliability gate below; the repair establishes interface
stability, not annotation validity.

In their matched 11,683-paper corpus, the human reference has normalized entropy 0.926 on the
opportunity axis and 0.920 on the method axis; the nine main LLM settings span 0.550--0.758 and
0.723--0.879, respectively. Their reasoning ablation is also relevant to experimental control:
Qwen3-8B thinking raises bridge mass from 49.7% to 71.1%, synthesis mass from 38.7% to 52.2%, and
human-opportunity TVD from 0.382 to 0.590. Our teacher, students, historical renderer, and annotator
were already frozen with thinking disabled before this paper-derived analysis was added. We cite
the ablation as a reason to report that control explicitly, not as post-hoc evidence favoring any
distillation method.

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
the same ordinal scale defined in `configs/evaluation/research_taste.yaml`. The compact SGLang
output grammar and the strict Pydantic validator in
`src/novelty_distill/evaluation/research_taste.py` fix field order, allowed labels, booleans, and
score ranges while preventing unconstrained whitespace loops.

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
