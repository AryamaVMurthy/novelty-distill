# Three-seed automatic descriptive diagnostics

Producer commit: `c480579dd9105429e85c293472530a21b595c6a0`

Judge distributions are automatic rubric diagnostics. Semantic curves are embedding-defined descriptive sensitivity analyses; without human boundary calibration they do not establish semantic equivalence, diversity, or novelty.

This report deliberately leaves the human-calibrated semantic boundary unresolved. All semantic tables below are threshold sensitivity diagnostics, not novelty scores.

## Judge score-distribution audit

Rates are computed over the 1,658 prompt-level K=4 means. A ceiling rate therefore means all four answers received 5/5 on that dimension for a prompt; it is not an individual-answer ceiling rate.

| Method | Feas. mean | Feas. ceiling | Sound. mean | Sound. ceiling | Relevance >=4.75 | Clarity >=4.75 | Compliance >=4.75 | Tokens | Length stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 (fixed) | 4.2176 | 0.1013 | 4.6025 | 0.3739 | 0.9349 | 0.9710 | 1.0000 | 329.0 | 0.000000 |
| A1 (fixed) | 4.6028 | 0.4071 | 4.9213 | 0.8281 | 0.9940 | 0.9970 | 1.0000 | 302.0 | 0.000000 |
| B2a | 3.3853 | 0.0008 | 3.1662 | 0.0010 | 0.0390 | 0.1313 | 0.9467 | 212.0 | 0.000050 |
| B2b | 3.4068 | 0.0024 | 3.1897 | 0.0024 | 0.0462 | 0.1504 | 0.9544 | 212.0 | 0.000000 |
| C1-best1 | 4.2447 | 0.1019 | 4.6625 | 0.4073 | 0.9493 | 0.9755 | 0.9996 | 348.3 | 0.000101 |
| C2-best1 | 4.2387 | 0.1078 | 4.6546 | 0.4262 | 0.9435 | 0.9793 | 1.0000 | 344.0 | 0.000050 |
| D1 | 4.2390 | 0.0963 | 4.6657 | 0.4198 | 0.9572 | 0.9811 | 1.0000 | 360.1 | 0.000151 |
| D2 | 4.2283 | 0.1021 | 4.6428 | 0.4041 | 0.9419 | 0.9753 | 1.0000 | 353.5 | 0.000151 |

Prompt-low rates use a K=4 prompt mean at or below 3/5. Length correlations are within-run Pearson correlations between the prompt's mean completion length and the prompt-level dimension mean.

| Method | Feasibility <=3 | Soundness <=3 | Feas.-length r | Sound.-length r |
|---|---:|---:|---:|---:|
| A0 (fixed) | 0.0054 | 0.0036 | 0.0757 | 0.0470 |
| A1 (fixed) | 0.0000 | 0.0000 | 0.1458 | 0.0781 |
| B2a | 0.3526 | 0.6064 | 0.0064 | 0.0248 |
| B2b | 0.3307 | 0.5814 | -0.0055 | 0.0110 |
| C1-best1 | 0.0058 | 0.0054 | 0.0862 | -0.0126 |
| C2-best1 | 0.0066 | 0.0038 | 0.0920 | 0.0096 |
| D1 | 0.0048 | 0.0044 | 0.1507 | -0.0175 |
| D2 | 0.0058 | 0.0034 | 0.1168 | -0.0109 |

## Quality-qualified semantic yield across all thresholds

Each trained entry is the mean of checkpoint seeds 17/29/43. A0/A1 are fixed seed-17 generation realizations.

| Threshold | A0 | A1 | B2a | B2b | C1-best1 | C2-best1 | D1 | D2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.700 | 0.9934 | 1.0000 | 0.3422 | 0.3613 | 0.9948 | 0.9930 | 0.9958 | 0.9948 |
| 0.750 | 0.9934 | 1.0006 | 0.3422 | 0.3613 | 0.9948 | 0.9930 | 0.9958 | 0.9948 |
| 0.800 | 0.9982 | 1.0078 | 0.3428 | 0.3615 | 0.9986 | 0.9940 | 1.0004 | 0.9964 |
| 0.820 | 1.0054 | 1.0145 | 0.3432 | 0.3623 | 1.0046 | 0.9958 | 1.0080 | 0.9994 |
| 0.850 | 1.0356 | 1.0458 | 0.3500 | 0.3699 | 1.0396 | 1.0147 | 1.0374 | 1.0155 |
| 0.900 | 1.2400 | 1.2793 | 0.3828 | 0.4063 | 1.3072 | 1.1596 | 1.3000 | 1.1594 |
| 0.940 | 1.8776 | 1.9620 | 0.4526 | 0.4861 | 2.1898 | 1.7549 | 2.1677 | 1.7228 |
| 0.950 | 2.1490 | 2.3179 | 0.4779 | 0.5179 | 2.5728 | 2.0611 | 2.5790 | 2.0283 |

## Contrast stability over the eight embedding thresholds

`Oriented range` is positive in the configured desirable direction (higher for coverage/recall/precision, lower for ClusterJSD). Direction is descriptive and does not make every teacher-fidelity metric scientific value.

| Metric | Contrast | Favorable thresholds | Direction | Oriented range |
|---|---|---:|---|---:|
| quality_qualified_semantic_yield | B2a-vs-A0 | 0/8 | unfavorable | [-1.6711, -0.6512] |
| student_semantic_clusters | B2a-vs-A0 | 8/8 | favorable | [0.0004, 0.8221] |
| teacher_mode_recall | B2a-vs-A0 | 0/8 | mixed | [-0.2284, 0.0000] |
| teacher_mode_precision | B2a-vs-A0 | 0/8 | unfavorable | [-0.2428, -0.0002] |
| cluster_jsd | B2a-vs-A0 | 0/8 | unfavorable | [-0.2409, -0.0001] |
| quality_adjusted_coverage | B2a-vs-A0 | 3/8 | mixed | [-0.1688, 0.1393] |
| quality_qualified_semantic_yield | B2b-vs-B2a | 8/8 | favorable | [0.0187, 0.0400] |
| student_semantic_clusters | B2b-vs-B2a | 4/8 | mixed | [-0.0092, 0.0155] |
| teacher_mode_recall | B2b-vs-B2a | 5/8 | mixed | [-0.0012, 0.0140] |
| teacher_mode_precision | B2b-vs-B2a | 6/8 | mixed | [-0.0007, 0.0143] |
| cluster_jsd | B2b-vs-B2a | 6/8 | mixed | [-0.0007, 0.0104] |
| quality_adjusted_coverage | B2b-vs-B2a | 7/8 | mixed | [-0.0000, 0.0262] |
| quality_qualified_semantic_yield | C1-vs-B2b | 8/8 | favorable | [0.6335, 2.0549] |
| student_semantic_clusters | C1-vs-B2b | 0/8 | unfavorable | [-0.4109, -0.0010] |
| teacher_mode_recall | C1-vs-B2b | 7/8 | mixed | [0.0000, 0.2435] |
| teacher_mode_precision | C1-vs-B2b | 8/8 | favorable | [0.0005, 0.2468] |
| cluster_jsd | C1-vs-B2b | 8/8 | favorable | [0.0002, 0.2524] |
| quality_adjusted_coverage | C1-vs-B2b | 8/8 | favorable | [0.0483, 0.2537] |
| quality_qualified_semantic_yield | C2-vs-C1 | 0/8 | unfavorable | [-0.5117, -0.0018] |
| student_semantic_clusters | C2-vs-C1 | 0/8 | unfavorable | [-0.5360, -0.0008] |
| teacher_mode_recall | C2-vs-C1 | 3/8 | mixed | [-0.0031, 0.0064] |
| teacher_mode_precision | C2-vs-C1 | 8/8 | favorable | [0.0004, 0.0315] |
| cluster_jsd | C2-vs-C1 | 8/8 | favorable | [0.0001, 0.0191] |
| quality_adjusted_coverage | C2-vs-C1 | 0/8 | unfavorable | [-0.5000, -0.0028] |
| quality_qualified_semantic_yield | D1-vs-C1 | 5/8 | mixed | [-0.0221, 0.0062] |
| student_semantic_clusters | D1-vs-C1 | 1/8 | mixed | [-0.0575, 0.0024] |
| teacher_mode_recall | D1-vs-C1 | 0/8 | mixed | [-0.0200, 0.0000] |
| teacher_mode_precision | D1-vs-C1 | 2/8 | mixed | [-0.0167, 0.0005] |
| cluster_jsd | D1-vs-C1 | 2/8 | mixed | [-0.0217, 0.0002] |
| quality_adjusted_coverage | D1-vs-C1 | 1/8 | mixed | [-0.0540, 0.0028] |
| quality_qualified_semantic_yield | D2-vs-C2 | 5/8 | mixed | [-0.0328, 0.0036] |
| student_semantic_clusters | D2-vs-C2 | 5/8 | mixed | [-0.0539, 0.0115] |
| teacher_mode_recall | D2-vs-C2 | 0/8 | mixed | [-0.0252, 0.0000] |
| teacher_mode_precision | D2-vs-C2 | 0/8 | unfavorable | [-0.0199, -0.0001] |
| cluster_jsd | D2-vs-C2 | 0/8 | unfavorable | [-0.0221, -0.0000] |
| quality_adjusted_coverage | D2-vs-C2 | 3/8 | mixed | [-0.0519, 0.0090] |
| quality_qualified_semantic_yield | D2-vs-D1 | 0/8 | unfavorable | [-0.5507, -0.0010] |
| student_semantic_clusters | D2-vs-D1 | 1/8 | mixed | [-0.5629, 0.0002] |
| teacher_mode_recall | D2-vs-D1 | 3/8 | mixed | [-0.0024, 0.0084] |
| teacher_mode_precision | D2-vs-D1 | 7/8 | mixed | [-0.0001, 0.0284] |
| cluster_jsd | D2-vs-D1 | 6/8 | mixed | [-0.0001, 0.0187] |
| quality_adjusted_coverage | D2-vs-D1 | 0/8 | unfavorable | [-0.5283, -0.0025] |

## Main descriptive findings

- B2a has more raw student embedding clusters than A0 at all eight thresholds, but lower quality-qualified yield at all eight. Its extra clusters therefore do not represent useful breadth under the automatic quality gate.
- C1 has fewer raw clusters than B2b at all eight thresholds but higher quality-qualified yield at all eight. The central hard-KD failure is quality, not a simple absence of string or embedding variation.
- C2 is below C1 on both raw clusters and quality-qualified yield at all eight thresholds, while having higher teacher-mode precision and lower ClusterJSD at all eight. In this embedding geometry, reverse KL is more teacher-concentrated and less broad.
- D1 versus C1 is mixed across thresholds on every semantic diagnostic; there is no stable automatic on-policy forward-KL advantage.
- D2 versus C2 is mixed on qualified yield and raw clusters, but has lower teacher-mode precision and higher ClusterJSD at every threshold. The current on-policy reverse-KL recipe does not improve teacher-distribution fidelity.
- D2 is below D1 on quality-qualified yield and quality-adjusted coverage at all eight thresholds. This is a robust automatic pattern, not a human-validated semantic-diversity conclusion.
- Prompt-level soundness-length correlations are near zero for all six trained methods. Feasibility-length correlations are small (largest for D1), so response length alone does not explain the hard-KD quality collapse.

## Interpretation boundary

- Stable behavior across thresholds is stronger descriptive evidence than an effect at 0.94 alone, but it still inherits the embedding model and K=4 sample.
- Relevance, clarity, and compliance remain strongly compressed; feasibility and soundness carry most of the automatic judge discrimination.
- No row establishes literature-grounded novelty or human semantic equivalence.
