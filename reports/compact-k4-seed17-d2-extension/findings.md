# Frozen TOMATO contrast findings

Producer Git commit: `453f075f4ef697e086504b86432c655eb15813c5`.

All estimates are treatment minus reference over paired prompts. Confidence intervals and sign-flip p-values use 10,000 Monte Carlo samples with seed 17; Holm correction is applied across every declared contrast within each metric. The 95% confidence intervals are pointwise, not simultaneous; multiplicity correction applies to p-values only. These are operational judge/embedding outcomes, not human-validated scientific novelty labels.

## Descriptive method levels

D2 is newly generated at K=4 with the same frozen decoding controls as the compact study. A0 reuses the deterministic first four samples from its validated K=16 source; A1 teacher modes use the first four samples from their validated K=16 source; every trained method uses four student samples. All extension contrasts are therefore sampling-budget matched at K=4.

ClusterJSD is a finite-sample plug-in estimate over eight teacher draws and the declared number of student draws. It is comparable only under the same sampling and clustering protocol, and is not an unbiased estimate of population divergence.

| Method | n | `cluster_jsd` mean | `completion_tokens_mean` mean | `length_stop_rate` mean | `nearest_training_target_similarity_mean` mean | `quality_adjusted_coverage` mean | `student_clarity_mean` mean | `student_feasibility_mean` mean | `student_instruction_compliance_mean` mean | `student_quality_mean` mean | `student_relevance_mean` mean | `student_semantic_clusters` mean | `student_soundness_mean` mean | `teacher_mode_precision` mean | `teacher_mode_recall` mean | `viable_semantic_yield` mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A0` | 1658 | 0.910177 | 328.993 | 0 | 0.791318 | 1.98423 | 4.96321 | 4.21758 | 4.9997 | 0.935502 | 4.92702 | 2.10676 | 4.60253 | 0.0867008 | 0.120376 | 1.89988 |
| `B1` | 1658 | 0.986141 | 344.059 | 0.0473462 | 0.761672 | 2.89147 | 4.31861 | 3.63571 | 4.65697 | 0.774495 | 4.27533 | 3.73945 | 3.60329 | 0.0103538 | 0.0250804 | 1.79916 |
| `B2b` | 1658 | 0.981044 | 212.417 | 0 | 0.790296 | 2.11071 | 4.22964 | 3.39837 | 4.94255 | 0.738405 | 4.02141 | 2.83353 | 3.17612 | 0.0169381 | 0.0306092 | 0.564536 |
| `C1-best1` | 1658 | 0.909733 | 347.154 | 0 | 0.792243 | 2.35042 | 4.96864 | 4.23854 | 4.9991 | 0.940282 | 4.94195 | 2.49095 | 4.65742 | 0.0839365 | 0.133645 | 2.24789 |
| `C2-best1` | 1658 | 0.903412 | 344.925 | 0 | 0.794402 | 1.89617 | 4.97497 | 4.2399 | 4.9997 | 0.940614 | 4.94195 | 2.0006 | 4.65576 | 0.0943908 | 0.134751 | 1.79493 |
| `D1` | 1658 | 0.923624 | 359.284 | 0 | 0.791693 | 2.28957 | 4.9712 | 4.23356 | 4.99955 | 0.940425 | 4.94587 | 2.42581 | 4.65832 | 0.0747386 | 0.11726 | 2.22256 |
| `D2` | 1658 | 0.913005 | 352.434 | 0.000150784 | 0.793561 | 1.82765 | 4.97075 | 4.2307 | 4.99985 | 0.938744 | 4.93562 | 1.92762 | 4.63797 | 0.0892139 | 0.125402 | 1.73703 |

## Primary-threshold paired estimates

### `student_feasibility_mean`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D2-vs-D1` | 1658 | 4.23356 | 4.2307 | -0.0028649 | [-0.0161339, 0.0104041] | -0.0104496 | 0.692031 | 0.692031 |
| `D2-vs-C2-best1` | 1658 | 4.2399 | 4.2307 | -0.00919783 | [-0.022316, 0.00331725] | -0.0344603 | 0.171483 | 0.514449 |
| `D2-vs-C1-best1` | 1658 | 4.23854 | 4.2307 | -0.00784077 | [-0.0215621, 0.00573356] | -0.0277514 | 0.269873 | 0.539746 |

### `student_soundness_mean`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D2-vs-D1` | 1658 | 4.65832 | 4.63797 | -0.0203559 | [-0.0354343, -0.00542823] | -0.0647282 | 0.00939906 | 0.0281972 |
| `D2-vs-C2-best1` | 1658 | 4.65576 | 4.63797 | -0.0177925 | [-0.0333233, -0.0028649] | -0.0558618 | 0.0262974 | 0.0383962 |
| `D2-vs-C1-best1` | 1658 | 4.65742 | 4.63797 | -0.0194511 | [-0.0354343, -0.00346803] | -0.0590062 | 0.0191981 | 0.0383962 |

### `teacher_mode_recall`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D2-vs-D1` | 1658 | 0.11726 | 0.125402 | 0.00814234 | [-0.004423, 0.0209087] | 0.0304902 | 0.212879 | 0.425757 |
| `D2-vs-C2-best1` | 1658 | 0.134751 | 0.125402 | -0.00934861 | [-0.0216626, 0.00271411] | -0.0364834 | 0.134487 | 0.40346 |
| `D2-vs-C1-best1` | 1658 | 0.133645 | 0.125402 | -0.00824286 | [-0.0213611, 0.00452478] | -0.0306569 | 0.220178 | 0.425757 |

### `viable_semantic_yield`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D2-vs-D1` | 1658 | 2.22256 | 1.73703 | -0.485525 | [-0.528347, -0.441496] | -0.537477 | 9.999e-05 | 0.00029997 |
| `D2-vs-C2-best1` | 1658 | 1.79493 | 1.73703 | -0.0579011 | [-0.0971049, -0.0180941] | -0.0704131 | 0.00469953 | 0.00469953 |
| `D2-vs-C1-best1` | 1658 | 2.24789 | 1.73703 | -0.510856 | [-0.553694, -0.466224] | -0.553737 | 9.999e-05 | 0.00029997 |

## Threshold-curve prompt directions

F/T/U is the number of paired prompts favorable, tied, or unfavorable for the treatment at that threshold. `cluster_jsd` is favorable when lower; the other declared threshold metrics are favorable when higher. Direction counts are descriptive and are not an additional multiplicity-adjusted hypothesis family. Here favorable means teacher-distribution fidelity or declared operational breadth, not that every unmatched student mode is scientifically invalid.

### `student_semantic_clusters`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `D2-vs-D1` | higher | mixed | 0.700: 1/1657/0; 0.750: 6/1643/9; 0.800: 27/1594/37; 0.820: 41/1526/91; 0.850: 106/1361/191; 0.900: 232/963/463; 0.940: 204/659/795; 0.950: 190/603/865 |
| `D2-vs-C2-best1` | higher | mixed | 0.700: 1/1657/0; 0.750: 6/1648/4; 0.800: 32/1598/28; 0.820: 50/1553/55; 0.850: 129/1411/118; 0.900: 289/1062/307; 0.940: 375/815/468; 0.950: 414/736/508 |
| `D2-vs-C1-best1` | higher | mixed | 0.700: 1/1656/1; 0.750: 2/1650/6; 0.800: 28/1593/37; 0.820: 46/1528/84; 0.850: 109/1359/190; 0.900: 227/939/492; 0.940: 180/637/841; 0.950: 182/570/906 |

### `teacher_mode_recall`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `D2-vs-D1` | higher | mixed | 0.700: 0/1658/0; 0.750: 0/1658/0; 0.800: 7/1643/8; 0.820: 12/1625/21; 0.850: 39/1573/46; 0.900: 142/1365/151; 0.940: 116/1448/94; 0.950: 72/1534/52 |
| `D2-vs-C2-best1` | higher | mixed | 0.700: 0/1658/0; 0.750: 1/1656/1; 0.800: 7/1643/8; 0.820: 7/1633/18; 0.850: 34/1585/39; 0.900: 119/1370/169; 0.940: 96/1446/116; 0.950: 53/1531/74 |
| `D2-vs-C1-best1` | higher | mixed | 0.700: 0/1658/0; 0.750: 0/1658/0; 0.800: 4/1646/8; 0.820: 9/1626/23; 0.850: 44/1564/50; 0.900: 145/1332/181; 0.940: 105/1429/124; 0.950: 62/1528/68 |

### `teacher_mode_precision`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `D2-vs-D1` | higher | mixed | 0.700: 0/1657/1; 0.750: 9/1643/6; 0.800: 36/1597/25; 0.820: 79/1526/53; 0.850: 186/1356/116; 0.900: 351/1068/239; 0.940: 160/1391/107; 0.950: 84/1518/56 |
| `D2-vs-C2-best1` | higher | unfavorable | 0.700: 0/1657/1; 0.750: 4/1649/5; 0.800: 27/1601/30; 0.820: 46/1554/58; 0.850: 119/1408/131; 0.900: 237/1128/293; 0.940: 131/1386/141; 0.950: 64/1512/82 |
| `D2-vs-C1-best1` | higher | mixed | 0.700: 1/1656/1; 0.750: 6/1650/2; 0.800: 33/1596/29; 0.820: 75/1522/61; 0.850: 187/1351/120; 0.900: 351/1030/277; 0.940: 160/1362/136; 0.950: 80/1506/72 |

### `cluster_jsd`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `D2-vs-D1` | lower | mixed | 0.700: 0/1657/1; 0.750: 9/1641/8; 0.800: 47/1572/39; 0.820: 99/1493/66; 0.850: 226/1265/167; 0.900: 427/905/326; 0.940: 170/1367/121; 0.950: 90/1508/60 |
| `D2-vs-C2-best1` | lower | unfavorable | 0.700: 0/1657/1; 0.750: 4/1648/6; 0.800: 37/1578/43; 0.820: 63/1518/77; 0.850: 152/1331/175; 0.900: 312/957/389; 0.940: 130/1361/167; 0.950: 69/1504/85 |
| `D2-vs-C1-best1` | lower | mixed | 0.700: 1/1656/1; 0.750: 7/1647/4; 0.800: 38/1576/44; 0.820: 92/1496/70; 0.850: 208/1272/178; 0.900: 402/889/367; 0.940: 157/1334/167; 0.950: 80/1500/78 |

### `quality_adjusted_coverage`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `D2-vs-D1` | higher | unfavorable | 0.700: 185/1230/243; 0.750: 190/1221/247; 0.800: 209/1174/275; 0.820: 214/1126/318; 0.850: 285/961/412; 0.900: 393/592/673; 0.940: 349/354/955; 0.950: 349/278/1031 |
| `D2-vs-C2-best1` | higher | mixed | 0.700: 208/1230/220; 0.750: 212/1224/222; 0.800: 239/1173/246; 0.820: 251/1137/270; 0.850: 322/1005/331; 0.900: 477/661/520; 0.940: 535/468/655; 0.950: 589/368/701 |
| `D2-vs-C1-best1` | higher | unfavorable | 0.700: 188/1223/247; 0.750: 188/1221/249; 0.800: 214/1169/275; 0.820: 222/1121/315; 0.850: 293/957/408; 0.900: 407/556/695; 0.940: 330/333/995; 0.950: 332/248/1078 |

### `viable_semantic_yield`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `D2-vs-D1` | higher | mixed | 0.700: 7/1648/3; 0.750: 7/1648/3; 0.800: 9/1636/13; 0.820: 13/1624/21; 0.850: 27/1571/60; 0.900: 102/1261/295; 0.940: 185/704/769; 0.950: 184/619/855 |
| `D2-vs-C2-best1` | higher | mixed | 0.700: 5/1650/3; 0.750: 5/1650/3; 0.800: 7/1646/5; 0.820: 11/1642/5; 0.850: 32/1597/29; 0.900: 152/1374/132; 0.940: 329/920/409; 0.950: 394/795/469 |
| `D2-vs-C1-best1` | higher | mixed | 0.700: 7/1648/3; 0.750: 7/1648/3; 0.800: 9/1638/11; 0.820: 15/1622/21; 0.850: 25/1573/60; 0.900: 104/1231/323; 0.940: 169/708/781; 0.950: 198/587/873 |
