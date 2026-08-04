# Frozen TOMATO contrast findings

Producer Git commit: `68d35683732075fa1ccb55f2b687026c055a3821`.

All estimates are treatment minus reference over paired prompts. Confidence intervals and sign-flip p-values use 10,000 Monte Carlo samples with seed 17; Holm correction is applied across every declared contrast within each metric. The 95% confidence intervals are pointwise, not simultaneous; multiplicity correction applies to p-values only. These are operational judge/embedding outcomes, not human-validated scientific novelty labels.

## Descriptive method levels

B2a is newly generated at K=4 with the same frozen decoding controls as the compact study. A0 reuses the deterministic first four samples from its validated K=16 source; A1 teacher modes use the first four samples from their validated K=16 source; every trained method uses four student samples. All extension contrasts are sampling-budget matched at K=4.

ClusterJSD is a finite-sample plug-in estimate over eight teacher draws and the declared number of student draws. It is comparable only under the same sampling and clustering protocol, and is not an unbiased estimate of population divergence.

Evidence policy: `posthoc-validity-quarantine-v1` (SHA-256 `04a07d4a54807cf3e391d4df1ab0e0b9a63fa0316b3a1b8ee59aa032f715aceb`).

Quarantined methods remain reproducibility and failure-analysis artifacts. They may appear in descriptive tables but cannot enter primary inferential contrasts until rerun on a reviewed, contamination-controlled historical-target artifact.

Semantic validity policy: `uncalibrated-semantic-threshold-quarantine-v1`.

Until the embedding equivalence threshold is calibrated against blinded human labels, teacher-mode recall, precision, ClusterJSD, quality-adjusted coverage, semantic cluster counts, and quality-qualified semantic yield are threshold-curve diagnostics only. They cannot enter a primary single-threshold inferential family.

| Method | Evidence status | n | `cluster_jsd` mean | `completion_tokens_mean` mean | `length_stop_rate` mean | `nearest_training_target_similarity_mean` mean | `quality_adjusted_coverage` mean | `quality_qualified_semantic_yield` mean | `student_clarity_mean` mean | `student_feasibility_mean` mean | `student_instruction_compliance_mean` mean | `student_quality_mean` mean | `student_relevance_mean` mean | `student_semantic_clusters` mean | `student_soundness_mean` mean | `teacher_mode_precision` mean | `teacher_mode_recall` mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A0` | primary_eligible | 1658 | 0.910177 | 328.993 | 0 | 0.791318 | 1.98423 | 1.87756 | 4.96321 | 4.21758 | 4.9997 | 0.935502 | 4.92702 | 2.10676 | 4.60253 | 0.0867008 | 0.120376 |
| `B2a` | primary_eligible | 1658 | 0.984153 | 212.745 | 0 | 0.790233 | 2.11345 | 0.452955 | 4.22482 | 3.39762 | 4.93637 | 0.736859 | 4.01297 | 2.84379 | 3.16541 | 0.0139224 | 0.0251307 |
| `B2b` | primary_eligible | 1658 | 0.981044 | 212.417 | 0 | 0.790296 | 2.11071 | 0.460193 | 4.22964 | 3.39837 | 4.94255 | 0.738405 | 4.02141 | 2.83353 | 3.17612 | 0.0169381 | 0.0306092 |
| `C1-best1` | primary_eligible | 1658 | 0.909733 | 347.154 | 0 | 0.792243 | 2.35042 | 2.22376 | 4.96864 | 4.23854 | 4.9991 | 0.940282 | 4.94195 | 2.49095 | 4.65742 | 0.0839365 | 0.133645 |

## Primary-threshold paired estimates

### `student_feasibility_mean`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B2a-vs-A0` | 1658 | 4.21758 | 3.39762 | -0.819964 | [-0.838209, -0.801568] | -2.18134 | 9.999e-05 | 0.00029997 |
| `B2b-vs-B2a` | 1658 | 3.39762 | 3.39837 | 0.00075392 | [-0.013269, 0.0144753] | 0.00259779 | 0.931707 | 0.931707 |
| `C1-best1-vs-B2a` | 1658 | 3.39762 | 4.23854 | 0.840923 | [0.82313, 0.858565] | 2.26647 | 9.999e-05 | 0.00029997 |

### `student_soundness_mean`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B2a-vs-A0` | 1658 | 4.60253 | 3.16541 | -1.43712 | [-1.45733, -1.41647] | -3.35792 | 9.999e-05 | 0.00029997 |
| `B2b-vs-B2a` | 1658 | 3.16541 | 3.17612 | 0.0107057 | [-0.00241631, 0.0239747] | 0.0392542 | 0.110789 | 0.110789 |
| `C1-best1-vs-B2a` | 1658 | 3.16541 | 4.65742 | 1.49201 | [1.47165, 1.51221] | 3.52515 | 9.999e-05 | 0.00029997 |

## Threshold-curve prompt directions

F/T/U is the number of paired prompts favorable, tied, or unfavorable for the treatment at that threshold. `cluster_jsd` is favorable when lower; the other declared threshold metrics are favorable when higher. Direction counts are descriptive and are not an additional multiplicity-adjusted hypothesis family. Here favorable means teacher-distribution fidelity or declared operational breadth, not that every unmatched student mode is scientifically invalid.

### `student_semantic_clusters`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B2a-vs-A0` | higher | favorable | 0.700: 4/1654/0; 0.750: 25/1628/5; 0.800: 166/1469/23; 0.820: 275/1334/49; 0.850: 518/1038/102; 0.900: 760/636/262; 0.940: 998/433/227; 0.950: 996/470/192 |
| `B2b-vs-B2a` | higher | mixed | 0.700: 3/1651/4; 0.750: 10/1637/11; 0.800: 64/1504/90; 0.820: 111/1430/117; 0.850: 202/1226/230; 0.900: 374/912/372; 0.940: 388/860/410; 0.950: 397/916/345 |
| `C1-best1-vs-B2a` | higher | unfavorable | 0.700: 1/1652/5; 0.750: 6/1627/25; 0.800: 32/1458/168; 0.820: 56/1329/273; 0.850: 121/1047/490; 0.900: 301/680/677; 0.940: 343/583/732; 0.950: 326/646/686 |

### `teacher_mode_recall`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B2a-vs-A0` | higher | mixed | 0.700: 0/1658/0; 0.750: 0/1656/2; 0.800: 8/1634/16; 0.820: 9/1604/45; 0.850: 43/1462/153; 0.900: 69/1086/503; 0.940: 23/1395/240; 0.950: 10/1532/116 |
| `B2b-vs-B2a` | higher | mixed | 0.700: 0/1658/0; 0.750: 2/1655/1; 0.800: 8/1640/10; 0.820: 25/1614/19; 0.850: 64/1541/53; 0.900: 153/1393/112; 0.940: 36/1593/29; 0.950: 9/1640/9 |
| `C1-best1-vs-B2a` | higher | mixed | 0.700: 0/1658/0; 0.750: 2/1656/0; 0.800: 22/1631/5; 0.820: 54/1598/6; 0.850: 162/1466/30; 0.900: 547/1050/61; 0.940: 262/1377/19; 0.950: 108/1543/7 |

### `teacher_mode_precision`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B2a-vs-A0` | higher | unfavorable | 0.700: 0/1654/4; 0.750: 4/1627/27; 0.800: 20/1465/173; 0.820: 40/1312/306; 0.850: 73/987/598; 0.900: 104/719/835; 0.940: 25/1371/262; 0.950: 11/1525/122 |
| `B2b-vs-B2a` | higher | mixed | 0.700: 4/1651/3; 0.750: 12/1637/9; 0.800: 88/1505/65; 0.820: 129/1418/111; 0.850: 237/1230/191; 0.900: 266/1172/220; 0.940: 47/1576/35; 0.950: 14/1631/13 |
| `C1-best1-vs-B2a` | higher | favorable | 0.700: 5/1652/1; 0.750: 27/1626/5; 0.800: 184/1455/19; 0.820: 310/1313/35; 0.850: 591/996/71; 0.900: 875/688/95; 0.940: 281/1358/19; 0.950: 113/1536/9 |

### `cluster_jsd`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B2a-vs-A0` | lower | unfavorable | 0.700: 0/1654/4; 0.750: 5/1624/29; 0.800: 26/1444/188; 0.820: 53/1269/336; 0.850: 97/914/647; 0.900: 132/642/884; 0.940: 24/1372/262; 0.950: 10/1525/123 |
| `B2b-vs-B2a` | lower | mixed | 0.700: 4/1650/4; 0.750: 20/1624/14; 0.800: 115/1459/84; 0.820: 183/1332/143; 0.850: 307/1099/252; 0.900: 298/1102/258; 0.940: 42/1577/39; 0.950: 12/1636/10 |
| `C1-best1-vs-B2a` | lower | favorable | 0.700: 5/1652/1; 0.750: 30/1622/6; 0.800: 197/1435/26; 0.820: 335/1281/42; 0.850: 653/927/78; 0.900: 946/613/99; 0.940: 277/1358/23; 0.950: 114/1535/9 |

### `quality_adjusted_coverage`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B2a-vs-A0` | higher | mixed | 0.700: 10/106/1542; 0.750: 31/105/1522; 0.800: 171/103/1384; 0.820: 277/87/1294; 0.850: 521/76/1061; 0.900: 749/23/886; 0.940: 926/35/697; 0.950: 914/32/712 |
| `B2b-vs-B2a` | higher | mixed | 0.700: 452/764/442; 0.750: 461/748/449; 0.800: 490/669/499; 0.820: 513/635/510; 0.850: 577/497/584; 0.900: 691/291/676; 0.940: 734/183/741; 0.950: 749/173/736 |
| `C1-best1-vs-B2a` | higher | favorable | 0.700: 1544/106/8; 0.750: 1524/106/28; 0.800: 1387/101/170; 0.820: 1295/91/272; 0.850: 1092/78/488; 0.900: 972/26/660; 0.940: 963/34/661; 0.950: 1027/35/596 |

### `quality_qualified_semantic_yield`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B2a-vs-A0` | higher | unfavorable | 0.700: 0/592/1066; 0.750: 0/592/1066; 0.800: 0/591/1067; 0.820: 0/590/1068; 0.850: 8/566/1084; 0.900: 36/453/1169; 0.940: 63/236/1359; 0.950: 65/190/1403 |
| `B2b-vs-B2a` | higher | mixed | 0.700: 187/1282/189; 0.750: 187/1282/189; 0.800: 188/1281/189; 0.820: 190/1279/189; 0.850: 195/1268/195; 0.900: 213/1223/222; 0.940: 261/1134/263; 0.950: 270/1115/273 |
| `C1-best1-vs-B2a` | higher | favorable | 0.700: 1069/589/0; 0.750: 1069/589/0; 0.800: 1069/589/0; 0.820: 1073/585/0; 0.850: 1095/555/8; 0.900: 1205/425/28; 0.940: 1413/199/46; 0.950: 1482/136/40 |
