# Distillation hypotheses and analysis contract

This document freezes the main mechanistic comparisons before the TOMATO-1k models are examined.
It distinguishes predictions motivated by prior work from findings produced by this repository.

## Primary outcomes

All comparisons use the same temporal-test prompt IDs and K=16 generation controls. The primary
prompt-level outcomes are judge quality and feasibility, teacher ModeRecall@16, ModePrecision@16,
ClusterJSD, quality-adjusted coverage, and nearest-training-target similarity. Semantic labels are
assigned by clustering each teacher/student sample set jointly; independently numbered clusters
are not comparable. The declared cosine threshold is 0.82, with the complete 0.70--0.95 curve
reported as a sensitivity analysis.

## Pre-run predictions

| Contrast | Mechanistic prediction | Evidence that would contradict it |
|---|---|---|
| B2a/B2b/B2c versus B3 | Training on four reusable teacher responses should recover more teacher modes than any single-response selection. Best-1 may raise quality while increasing training-target similarity. | No ModeRecall or cluster-count gain for B3 across the threshold curve. |
| B3 versus B4 | With identical diverse-4 rows, GEM should reduce SFT overfitting and retain more generation diversity than cross-entropy SFT. | B4 has no coverage/JSD advantage, or gains only lexical uniqueness without semantic modes. |
| C1 versus C2 | Forward KL is expected to be more mode-covering; reverse KL is expected to be more mode-seeking, potentially trading recall for precision/quality. | Reverse KL improves recall and JSD without a precision/quality trade-off, or forward KL is more collapsed. |
| C1 static versus D1 on-policy | Student rollouts should reduce train/inference exposure mismatch, improving test quality or teacher-distribution alignment. | D1 fails to improve quality/JSD and merely increases training-target similarity. |
| D1 versus D2 versus D3 | Generalized JSD should lie between forward- and reverse-KL behavior; the direction and size are empirical. | A result at only one clustering threshold is not sufficient evidence. |
| C3 versus C1/C2 | DistiLLM's skew loss and adaptive replay may offer a better efficiency/quality compromise, but its shorter hardware-feasible context is a confound that must be reported. | Any comparison that hides truncation or unequal effective examples is invalid. |
| E2 versus E4 | Privileged historical hypotheses/inspirations should make the self-teacher's token feedback more useful, increasing quality/feasibility. | E2 matches E4 or only copies training targets more closely. |
| E2 versus E3 | Reverse-KL OPSD may be more selective and mode-seeking than forward-KL OPSD. | E3 increases coverage robustly without the expected selectivity trade-off. |

The forward/reverse-KL prediction is an inference from divergence geometry and the MiniLLM/GKD
literature; it is not assumed to hold automatically at the sequence-semantic level. GKD motivates
the exposure-mismatch contrast, GEM directly motivates the B3/B4 isolation, DistiLLM motivates the
skew-loss/adaptive-replay comparison, and OPSD motivates the privileged-context comparison.

## Analysis rules

- Aggregate at the prompt level; K=16 samples are repeated observations, not 16 independent tasks.
- Report paired bootstrap confidence intervals and standardized paired effects for promoted
  contrasts. Adjust confirmatory contrast p-values with Holm's method.
- Treat quality and coverage as separate axes. Do not claim a diversity win from unique strings,
  cluster count alone, or a quality-adjusted-coverage increase caused only by judge score.
- Report the full clustering threshold curve and per-prompt direction counts. A primary-threshold
  effect isolated to one prompt is exploratory.
- Report length-stop rate, completion-token distribution, and training-target similarity beside
  every headline comparison to expose truncation and memorization.
- The 1k gate fixes optimizer exposure at 1,000 target rows. `diverse4` expands to 4,000 available
  rows, so B3/B4 at this gate are budget-matched samples from the four-reference pool, not a full
  four-pass exposure. Report both `training_rows` and `optimizer_example_exposures`; any promoted
  four-reference finding requires a full-pool exposure sensitivity with repeated single-target
  controls at the same 4,000-row budget.
- Keep reverse-KL/JSD and optional controls at one training seed until the pipeline passes; promote
  central comparisons to seeds 17, 29, and 43 only after the frozen 1k gate.
- Do not pool results across the 4B/14B and 1.7B/8B model pairs. The latter is a replication, not
  an extra seed.

## Interpretation risks

The automatic judge and embedding model are fixed proxies, connected-component clustering can
chain samples, and teacher outputs are not ground-truth scientific modes. The paired concise
teacher calibration already showed that better feasibility can coexist with fewer measured modes.
Accordingly, the final report must preserve raw prompt-level metrics and seek expert annotation for
any central semantic-coverage claim.

## Primary source log

- Agarwal et al., *On-Policy Distillation of Language Models* (GKD), ICLR 2024:
  <https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf>
- Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*, ICLR 2024:
  <https://proceedings.iclr.cc/paper_files/paper/2024/hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html>
- Li et al., *Preserving Diversity in Supervised Fine-Tuning of Large Language Models*, ICLR 2025:
  <https://openreview.net/forum?id=NQEe7B7bSw>
- Ko et al., *DistiLLM: Towards Streamlined Distillation for Large Language Models*, ICML 2024:
  <https://openreview.net/forum?id=lsHZNNoC7r>
- Zhao et al., *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*:
  <https://arxiv.org/abs/2601.18734>
