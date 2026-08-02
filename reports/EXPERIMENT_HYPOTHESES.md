# Distillation hypotheses and analysis contract

This document freezes the main mechanistic comparisons before the TOMATO-1k models are examined.
It distinguishes predictions motivated by prior work from findings produced by this repository.

## Primary outcomes

All comparisons use the same temporal-test prompt IDs and K=16 generation controls. The primary
prompt-level outcomes are judge quality and feasibility, teacher ModeRecall@16, ModePrecision@16,
ClusterJSD, quality-adjusted coverage, and nearest-training-target similarity. Teacher samples are
clustered first to freeze one teacher-mode partition per prompt. For each student sample, every
teacher mode is scored by its minimum cosine similarity to any member; the sample joins the
best-scoring mode only when that complete-link score reaches the threshold. Unmatched students are
clustered as novel student modes. This prevents a student bridge from merging teacher modes or
violating the within-mode boundary differently by method. Before any student evaluation, the
production teacher-only gate showed that the original
connected-component threshold 0.82 collapsed to 1.001 modes per prompt. Moving connected
components to 0.95 yielded 3.558 modes but a deterministic qualitative audit found bridge-induced
phase behavior and obvious paraphrase splitting. The frozen primary partition therefore uses
complete linkage at cosine 0.94: all members of a mode must meet the threshold pairwise. It yields
3.798 teacher modes per prompt (median 4) while grouping the clearest duplicate hypotheses. The
complete eight-point 0.70--0.95 curve is still reported as a sensitivity analysis.

## Pre-run predictions

| Contrast | Mechanistic prediction | Evidence that would contradict it |
|---|---|---|
| B2a/B2b/B2c versus B3 | Training on four reusable teacher responses should recover more teacher modes than any single-response selection. Best-1 may raise quality while increasing training-target similarity. | No ModeRecall or cluster-count gain for B3 across the threshold curve. |
| B3 versus B4 | With identical diverse-4 rows, GEM should reduce SFT overfitting and retain more generation diversity than cross-entropy SFT. | B4 has no coverage/JSD advantage, or gains only lexical uniqueness without semantic modes. |
| C1 versus C2 | As a finite-step heuristic, forward KL may emphasize teacher-head coverage while reverse KL may be more selective, potentially trading recall for precision/quality. The direction is exploratory rather than a consequence guaranteed by KL geometry. | Reverse KL improves recall and JSD without a precision/quality trade-off, or forward KL is more collapsed. |
| C1 static versus D1 on-policy | Student rollouts should reduce train/inference exposure mismatch, improving test quality or teacher-distribution alignment. | D1 fails to improve quality/JSD and merely increases training-target similarity. |
| D1 versus D2 versus D3 | Generalized JSD should lie between forward- and reverse-KL behavior; the direction and size are empirical. | A result at only one clustering threshold is not sufficient evidence. |
| C3 versus C1/C2 | DistiLLM's skew loss and adaptive replay may offer a better efficiency/quality compromise, but its shorter hardware-feasible context is a confound that must be reported. | Any comparison that hides truncation or unequal effective examples is invalid. |
| E2 versus E4 | Privileged historical hypotheses/inspirations should make the self-teacher's token feedback more useful, increasing quality/feasibility. | E2 matches E4 or only copies training targets more closely. |
| E2 versus E3 | Reverse-KL OPSD may be more selective and mode-seeking than forward-KL OPSD. | E3 increases coverage robustly without the expected selectivity trade-off. |

The forward/reverse-KL prediction is a finite-optimization heuristic motivated by the MiniLLM/GKD
literature; it is not assumed to hold automatically at the sequence-semantic level. Wu et al.
specifically challenge a blanket mode-seeking/mode-covering interpretation for LLM distillation and
find different early emphasis on distribution tails versus heads instead. GKD motivates the
exposure-mismatch contrast, GEM directly motivates the B3/B4 isolation, DistiLLM motivates the
skew-loss/adaptive-replay comparison, and OPSD motivates the privileged-context comparison.

## Analysis rules

- Aggregate at the prompt level; K=16 samples are repeated observations, not 16 independent tasks.
- Report paired bootstrap confidence intervals and standardized paired effects for promoted
  contrasts. Adjust confirmatory contrast p-values with Holm's method. The reported 95% bootstrap
  intervals are pointwise rather than simultaneous confidence intervals; the Holm adjustment
  applies to p-values only.
- Holm adjustment is applied to all 17 declared contrasts separately within each metric, not to
  the complete cross-metric set. A favorable result selected after scanning multiple outcomes is
  exploratory unless its metric and contrast were declared as the specific claim in advance.
- A non-significant difference is not evidence of equivalence. Any later equivalence claim needs a
  scientifically justified smallest effect size of interest and a confidence interval contained
  inside those bounds; this 1k gate does not define such a bound post hoc.
- Treat quality and coverage as separate axes. Do not claim a diversity win from unique strings,
  cluster count alone, or a quality-adjusted-coverage increase caused only by judge score.
- Teacher-mode precision and low JSD are teacher-distribution fidelity outcomes, not monotonic
  scientific-value measures. A valid mode absent from eight teacher samples lowers precision just
  as an irrelevant mode does; interpret it beside judged quality, student-only mode count, and
  expert review rather than calling every precision decrease a novelty failure.
- ClusterJSD is the categorical plug-in divergence of the observed teacher/student mode counts.
  With eight teacher and sixteen student samples it is a finite-sample, protocol-dependent index,
  not an unbiased population divergence. Compare it only across methods with the same sampling and
  clustering contract; A3's K=1 value is descriptive only.
- Report the full clustering threshold curve and per-prompt direction counts. A primary-threshold
  effect isolated to one prompt is exploratory.
- Complete-linkage 0.94 is a teacher-only, pre-student calibration choice, not a universal
  semantic-equivalence boundary. Raw versus instructed embeddings can disagree materially, so
  any promoted result must be directionally stable across the curve and survive expert review.
- Report length-stop rate, completion-token distribution, and training-target similarity beside
  every headline comparison to expose truncation and memorization.
- Report each run's training-context audit. GKD preserves the full prompt and truncates only the
  completion tail when required; its frozen 2,048-token limit affects 3/1,000 human targets and no
  teacher-sampled/on-policy rows. SFT/OPSD use 3,072 with zero audited overflow. C3's 896-token cap
  remains a declared hardware confound rather than being normalized away after results.
- Use Qwen non-thinking chat rendering for every GKD prompt, matching teacher generation and
  evaluation; do not interpret a thinking-mode mismatch as evidence for on-policy distillation.
- Admit C3 as an adaptive-replay treatment only when its official log contains every requested
  optimizer step and all ten scheduled validation stages. Report the validation-loss trajectory
  and terminal student-replay probability; a zero terminal value is an empirical scheduler outcome,
  while a disabled scheduler is an invalid implementation.
- The 1k gate fixes optimizer exposure at 1,000 target rows. `diverse4` expands to 4,000 available
  rows, so B3/B4 at this gate are budget-matched samples from the four-reference pool, not a full
  four-pass exposure. Report both `training_rows` and `optimizer_example_exposures`; any promoted
  four-reference finding requires a full-pool exposure sensitivity with repeated single-target
  controls at the same 4,000-row budget.
- Keep reverse-KL/JSD and optional controls at one training seed until the pipeline passes; promote
  central comparisons to seeds 17, 29, and 43 only after the frozen 1k gate.
- Do not pool results across the 4B/14B and 1.7B/8B model pairs. The latter is a replication, not
  an extra seed.
- A3 contains one historical author response per prompt, whereas A0, A1, and trained generative
  methods use K=16. Its quality and feasibility remain useful author-anchored controls, but its
  semantic breadth, mode recall, precision, and JSD are not sampling-budget-matched comparisons.

## Interpretation risks

The automatic judge and embedding model are fixed proxies, complete linkage can split semantic
paraphrases near its boundary, and teacher outputs are not ground-truth scientific modes. The paired concise
teacher calibration already showed that better feasibility can coexist with fewer measured modes.
Accordingly, the final report must preserve raw prompt-level metrics and seek expert annotation for
any central semantic-coverage claim.

## Secondary research-taste analysis

After training began but before any temporal student metric was available, Chen, Zhao, and Cohan's
2026 preprint on human--LLM research-taste gaps motivated a separate descriptive analysis. It does
not modify the primary outcomes or declared contrast family. The analysis labels each idea by its
opportunity pattern and method paradigm, then reports entropy and TVD/JSD from the A3 human and A1
teacher distributions. Both all-K and deterministic sample-zero views are required. Automatic
results cannot support a headline claim until the independent human-agreement gate passes. The
complete frozen contract, attribution, and limitations are in
`reports/RESEARCH_TASTE_PROTOCOL.md`.

## Primary source log

- Agarwal et al., *On-Policy Distillation of Language Models* (GKD), ICLR 2024:
  <https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf>
- Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*, ICLR 2024:
  <https://proceedings.iclr.cc/paper_files/paper/2024/hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html>
- Wu et al., *Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language
  Models*, COLING 2025: <https://aclanthology.org/2025.coling-main.383/>
- Le Bronnec et al., *Exploring Precision and Recall to Assess the Quality and Diversity of LLMs*,
  ACL 2024: <https://aclanthology.org/2024.acl-long.616/>
- Li et al., *Preserving Diversity in Supervised Fine-Tuning of Large Language Models*, ICLR 2025:
  <https://openreview.net/forum?id=NQEe7B7bSw>
- Ko et al., *DistiLLM: Towards Streamlined Distillation for Large Language Models*, ICML 2024:
  <https://openreview.net/forum?id=lsHZNNoC7r>
- Zhao et al., *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*:
  <https://arxiv.org/abs/2601.18734>
- Chen et al., *Measuring the Gap Between Human and LLM Research Ideas*, arXiv preprint, 2026:
  <https://arxiv.org/abs/2607.01233>
