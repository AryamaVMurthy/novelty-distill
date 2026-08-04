# Frozen TOMATO contrast findings

> **Historical automatic analysis; interpretation superseded on 2026-08-05.**
> `B1` is quarantined, semantic single-threshold inference is disallowed, and
> `viable_semantic_yield` is legacy-only because it omitted feasibility. Use
> [`../EVALUATION_VALIDITY_AUDIT_20260805.md`](../EVALUATION_VALIDITY_AUDIT_20260805.md)
> and [`../audits/corrected-k4-seed17-v2-summary.json`](../audits/corrected-k4-seed17-v2-summary.json).

Producer Git commit: `c26a5dc1593b0d78fffe19637f7716574df29149`.

All estimates are treatment minus reference over paired prompts. Confidence intervals and sign-flip p-values use 10,000 Monte Carlo samples with seed 17; Holm correction is applied across every declared contrast within each metric. The 95% confidence intervals are pointwise, not simultaneous; multiplicity correction applies to p-values only. These are operational judge/embedding outcomes, not human-validated scientific novelty labels.

## Descriptive method levels

A0 reuses the deterministic first four samples from its validated K=16 source; every trained method is newly generated at K=4 with the same decoding controls. A1 teacher modes likewise use the first four samples from its validated K=16 source. All declared contrasts are therefore sampling-budget matched at K=4.

ClusterJSD is a finite-sample plug-in estimate over eight teacher draws and the declared number of student draws. It is comparable only under the same sampling and clustering protocol, and is not an unbiased estimate of population divergence.

| Method | n | `cluster_jsd` mean | `completion_tokens_mean` mean | `length_stop_rate` mean | `nearest_training_target_similarity_mean` mean | `quality_adjusted_coverage` mean | `student_clarity_mean` mean | `student_feasibility_mean` mean | `student_instruction_compliance_mean` mean | `student_quality_mean` mean | `student_relevance_mean` mean | `student_semantic_clusters` mean | `student_soundness_mean` mean | `teacher_mode_precision` mean | `teacher_mode_recall` mean | `viable_semantic_yield` mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A0` | 1658 | 0.910177 | 328.993 | 0 | 0.791318 | 1.98423 | 4.96321 | 4.21758 | 4.9997 | 0.935502 | 4.92702 | 2.10676 | 4.60253 | 0.0867008 | 0.120376 | 1.89988 |
| `B1` | 1658 | 0.986141 | 344.059 | 0.0473462 | 0.761672 | 2.89147 | 4.31861 | 3.63571 | 4.65697 | 0.774495 | 4.27533 | 3.73945 | 3.60329 | 0.0103538 | 0.0250804 | 1.79916 |
| `B2b` | 1658 | 0.981044 | 212.417 | 0 | 0.790296 | 2.11071 | 4.22964 | 3.39837 | 4.94255 | 0.738405 | 4.02141 | 2.83353 | 3.17612 | 0.0169381 | 0.0306092 | 0.564536 |
| `C1-best1` | 1658 | 0.909733 | 347.154 | 0 | 0.792243 | 2.35042 | 4.96864 | 4.23854 | 4.9991 | 0.940282 | 4.94195 | 2.49095 | 4.65742 | 0.0839365 | 0.133645 | 2.24789 |
| `C2-best1` | 1658 | 0.903412 | 344.925 | 0 | 0.794402 | 1.89617 | 4.97497 | 4.2399 | 4.9997 | 0.940614 | 4.94195 | 2.0006 | 4.65576 | 0.0943908 | 0.134751 | 1.79493 |
| `D1` | 1658 | 0.923624 | 359.284 | 0 | 0.791693 | 2.28957 | 4.9712 | 4.23356 | 4.99955 | 0.940425 | 4.94587 | 2.42581 | 4.65832 | 0.0747386 | 0.11726 | 2.22256 |

## Primary-threshold paired estimates

### `student_feasibility_mean`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B1-vs-A0` | 1658 | 4.21758 | 3.63571 | -0.581876 | [-0.603589, -0.56031] | -1.29955 | 9.999e-05 | 0.00049995 |
| `B2b-vs-B1` | 1658 | 3.63571 | 3.39837 | -0.237334 | [-0.258444, -0.216978] | -0.546413 | 9.999e-05 | 0.00049995 |
| `C1-best1-vs-B2b` | 1658 | 3.39837 | 4.23854 | 0.840169 | [0.822071, 0.857961] | 2.26176 | 9.999e-05 | 0.00049995 |
| `C2-best1-vs-C1-best1` | 1658 | 4.23854 | 4.2399 | 0.00135706 | [-0.0126659, 0.01538] | 0.00473364 | 0.865313 | 0.926107 |
| `D1-vs-C1-best1` | 1658 | 4.23854 | 4.23356 | -0.00497587 | [-0.0179433, 0.00814234] | -0.0185368 | 0.463054 | 0.926107 |

### `student_soundness_mean`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B1-vs-A0` | 1658 | 4.60253 | 3.60329 | -0.999246 | [-1.02488, -0.973462] | -1.88111 | 9.999e-05 | 0.00049995 |
| `B2b-vs-B1` | 1658 | 3.60329 | 3.17612 | -0.427171 | [-0.451146, -0.4038] | -0.860578 | 9.999e-05 | 0.00049995 |
| `C1-best1-vs-B2b` | 1658 | 3.17612 | 4.65742 | 1.4813 | [1.4608, 1.50136] | 3.49419 | 9.999e-05 | 0.00049995 |
| `C2-best1-vs-C1-best1` | 1658 | 4.65742 | 4.65576 | -0.00165862 | [-0.0170386, 0.0138721] | -0.0050848 | 0.845615 | 1 |
| `D1-vs-C1-best1` | 1658 | 4.65742 | 4.65832 | 0.000904704 | [-0.0143245, 0.0161339] | 0.00289008 | 0.923408 | 1 |

### `teacher_mode_recall`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B1-vs-A0` | 1658 | 0.120376 | 0.0250804 | -0.0952955 | [-0.109721, -0.0813216] | -0.322562 | 9.999e-05 | 0.00049995 |
| `B2b-vs-B1` | 1658 | 0.0250804 | 0.0306092 | 0.00552875 | [-0.00361882, 0.0147266] | 0.028533 | 0.250475 | 0.50095 |
| `C1-best1-vs-B2b` | 1658 | 0.0306092 | 0.133645 | 0.103036 | [0.0887113, 0.117561] | 0.339456 | 9.999e-05 | 0.00049995 |
| `C2-best1-vs-C1-best1` | 1658 | 0.133645 | 0.134751 | 0.00110575 | [-0.0113591, 0.0139739] | 0.00410492 | 0.879612 | 0.879612 |
| `D1-vs-C1-best1` | 1658 | 0.133645 | 0.11726 | -0.0163852 | [-0.0294544, -0.00306594] | -0.0589343 | 0.0161984 | 0.0485951 |

### `viable_semantic_yield`

| Contrast | n | Reference mean | Treatment mean | Mean difference | 95% CI | Cohen's dz | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B1-vs-A0` | 1658 | 1.89988 | 1.79916 | -0.100724 | [-0.171894, -0.0283474] | -0.0677892 | 0.0059994 | 0.0119988 |
| `B2b-vs-B1` | 1658 | 1.79916 | 0.564536 | -1.23462 | [-1.29011, -1.17913] | -1.06241 | 9.999e-05 | 0.00049995 |
| `C1-best1-vs-B2b` | 1658 | 0.564536 | 2.24789 | 1.68335 | [1.62182, 1.74487] | 1.31032 | 9.999e-05 | 0.00049995 |
| `C2-best1-vs-C1-best1` | 1658 | 2.24789 | 1.79493 | -0.452955 | [-0.495778, -0.410118] | -0.501297 | 9.999e-05 | 0.00049995 |
| `D1-vs-C1-best1` | 1658 | 2.24789 | 2.22256 | -0.0253317 | [-0.0681544, 0.0162847] | -0.0288338 | 0.243676 | 0.243676 |

## Threshold-curve prompt directions

F/T/U is the number of paired prompts favorable, tied, or unfavorable for the treatment at that threshold. `cluster_jsd` is favorable when lower; the other declared threshold metrics are favorable when higher. Direction counts are descriptive and are not an additional multiplicity-adjusted hypothesis family. Here favorable means teacher-distribution fidelity or declared operational breadth, not that every unmatched student mode is scientifically invalid.

### `student_semantic_clusters`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B1-vs-A0` | higher | favorable | 0.700: 18/1640/0; 0.750: 89/1565/4; 0.800: 345/1298/15; 0.820: 525/1109/24; 0.850: 855/771/32; 0.900: 1326/279/53; 0.940: 1477/166/15; 0.950: 1411/242/5 |
| `B2b-vs-B1` | higher | unfavorable | 0.700: 3/1637/18; 0.750: 15/1561/82; 0.800: 68/1293/297; 0.820: 116/1109/433; 0.850: 165/837/656; 0.900: 133/450/1075; 0.940: 64/528/1066; 0.950: 39/747/872 |
| `C1-best1-vs-B2b` | higher | unfavorable | 0.700: 1/1653/4; 0.750: 7/1627/24; 0.800: 30/1487/141; 0.820: 57/1334/267; 0.850: 119/1066/473; 0.900: 310/677/671; 0.940: 339/611/708; 0.950: 308/638/712 |
| `C2-best1-vs-C1-best1` | higher | unfavorable | 0.700: 0/1657/1; 0.750: 3/1646/9; 0.800: 27/1589/42; 0.820: 53/1520/85; 0.850: 95/1374/189; 0.900: 208/985/465; 0.940: 213/637/808; 0.950: 189/610/859 |
| `D1-vs-C1-best1` | higher | mixed | 0.700: 0/1657/1; 0.750: 8/1642/8; 0.800: 33/1593/32; 0.820: 72/1520/66; 0.850: 151/1359/148; 0.900: 325/974/359; 0.940: 411/738/509; 0.950: 410/770/478 |

### `teacher_mode_recall`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B1-vs-A0` | higher | mixed | 0.700: 0/1658/0; 0.750: 1/1656/1; 0.800: 11/1626/21; 0.820: 13/1585/60; 0.850: 41/1444/173; 0.900: 49/1109/500; 0.940: 21/1394/243; 0.950: 7/1530/121 |
| `B2b-vs-B1` | higher | mixed | 0.700: 0/1658/0; 0.750: 1/1655/2; 0.800: 15/1628/15; 0.820: 46/1582/30; 0.850: 131/1428/99; 0.900: 245/1222/191; 0.940: 53/1565/40; 0.950: 17/1630/11 |
| `C1-best1-vs-B2b` | higher | mixed | 0.700: 0/1658/0; 0.750: 1/1657/0; 0.800: 26/1626/6; 0.820: 50/1601/7; 0.850: 152/1474/32; 0.900: 511/1087/60; 0.940: 264/1365/29; 0.950: 110/1539/9 |
| `C2-best1-vs-C1-best1` | higher | mixed | 0.700: 0/1658/0; 0.750: 1/1656/1; 0.800: 5/1645/8; 0.820: 8/1639/11; 0.850: 41/1575/42; 0.900: 153/1362/143; 0.940: 125/1409/124; 0.950: 65/1543/50 |
| `D1-vs-C1-best1` | higher | mixed | 0.700: 0/1658/0; 0.750: 0/1658/0; 0.800: 2/1651/5; 0.820: 7/1639/12; 0.850: 43/1571/44; 0.900: 147/1334/177; 0.940: 96/1426/136; 0.950: 42/1546/70 |

### `teacher_mode_precision`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B1-vs-A0` | higher | unfavorable | 0.700: 0/1640/18; 0.750: 4/1564/90; 0.800: 20/1294/344; 0.820: 31/1100/527; 0.850: 60/758/840; 0.900: 58/609/991; 0.940: 21/1365/272; 0.950: 7/1526/125 |
| `B2b-vs-B1` | higher | favorable | 0.700: 18/1637/3; 0.750: 81/1561/16; 0.800: 288/1295/75; 0.820: 416/1109/133; 0.850: 582/844/232; 0.900: 501/903/254; 0.940: 63/1555/40; 0.950: 18/1629/11 |
| `C1-best1-vs-B2b` | higher | favorable | 0.700: 4/1653/1; 0.750: 25/1626/7; 0.800: 159/1477/22; 0.820: 300/1321/37; 0.850: 565/1016/77; 0.900: 864/693/101; 0.940: 278/1352/28; 0.950: 117/1531/10 |
| `C2-best1-vs-C1-best1` | higher | favorable | 0.700: 1/1657/0; 0.750: 8/1647/3; 0.800: 36/1594/28; 0.820: 81/1522/55; 0.850: 190/1359/109; 0.900: 358/1077/223; 0.940: 173/1347/138; 0.950: 85/1514/59 |
| `D1-vs-C1-best1` | higher | mixed | 0.700: 1/1657/0; 0.750: 8/1642/8; 0.800: 28/1595/35; 0.820: 64/1519/75; 0.850: 153/1356/149; 0.900: 285/1043/330; 0.940: 121/1375/162; 0.950: 51/1526/81 |

### `cluster_jsd`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B1-vs-A0` | lower | unfavorable | 0.700: 0/1640/18; 0.750: 5/1564/89; 0.800: 28/1281/349; 0.820: 45/1078/535; 0.850: 78/735/845; 0.900: 89/611/958; 0.940: 22/1371/265; 0.950: 7/1528/123 |
| `B2b-vs-B1` | lower | favorable | 0.700: 18/1636/4; 0.750: 82/1559/17; 0.800: 292/1274/92; 0.820: 433/1055/170; 0.850: 585/763/310; 0.900: 455/881/322; 0.940: 58/1559/41; 0.950: 17/1628/13 |
| `C1-best1-vs-B2b` | lower | favorable | 0.700: 4/1653/1; 0.750: 27/1624/7; 0.800: 173/1458/27; 0.820: 316/1293/49; 0.850: 630/939/89; 0.900: 933/622/103; 0.940: 286/1341/31; 0.950: 115/1533/10 |
| `C2-best1-vs-C1-best1` | lower | favorable | 0.700: 1/1657/0; 0.750: 10/1645/3; 0.800: 45/1575/38; 0.820: 98/1489/71; 0.850: 218/1283/157; 0.900: 429/905/324; 0.940: 195/1303/160; 0.950: 87/1504/67 |
| `D1-vs-C1-best1` | lower | mixed | 0.700: 1/1657/0; 0.750: 10/1640/8; 0.800: 36/1571/51; 0.820: 77/1486/95; 0.850: 193/1253/212; 0.900: 350/889/419; 0.940: 126/1344/188; 0.950: 54/1517/87 |

### `quality_adjusted_coverage`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B1-vs-A0` | higher | mixed | 0.700: 117/578/963; 0.750: 183/562/913; 0.800: 402/490/766; 0.820: 567/440/651; 0.850: 866/293/499; 0.900: 1282/72/304; 0.940: 1364/26/268; 0.950: 1295/29/334 |
| `B2b-vs-B1` | higher | unfavorable | 0.700: 121/323/1214; 0.750: 125/304/1229; 0.800: 159/251/1248; 0.820: 213/203/1242; 0.850: 253/140/1265; 0.900: 236/47/1375; 0.940: 233/44/1381; 0.950: 275/56/1327 |
| `C1-best1-vs-B2b` | higher | favorable | 0.700: 1547/102/9; 0.750: 1527/102/29; 0.800: 1416/97/145; 0.820: 1298/92/268; 0.850: 1119/65/474; 0.900: 969/29/660; 0.940: 979/32/647; 0.950: 996/41/621 |
| `C2-best1-vs-C1-best1` | higher | unfavorable | 0.700: 192/1215/251; 0.750: 196/1206/256; 0.800: 211/1163/284; 0.820: 227/1118/313; 0.850: 274/966/418; 0.900: 382/581/695; 0.940: 364/325/969; 0.950: 355/256/1047 |
| `D1-vs-C1-best1` | higher | mixed | 0.700: 199/1247/212; 0.750: 207/1235/216; 0.800: 233/1188/237; 0.820: 266/1122/270; 0.850: 351/964/343; 0.900: 530/556/572; 0.940: 614/321/723; 0.950: 646/298/714 |

### `viable_semantic_yield`

| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |
|---|---|---|---|
| `B1-vs-A0` | higher | mixed | 0.700: 2/1398/258; 0.750: 9/1391/258; 0.800: 76/1318/264; 0.820: 129/1257/272; 0.850: 254/1103/301; 0.900: 592/666/400; 0.940: 599/396/663; 0.950: 549/363/746 |
| `B2b-vs-B1` | higher | unfavorable | 0.700: 37/874/747; 0.750: 37/872/749; 0.800: 38/839/781; 0.820: 40/808/810; 0.850: 48/733/877; 0.900: 62/497/1099; 0.940: 75/394/1189; 0.950: 79/379/1200 |
| `C1-best1-vs-B2b` | higher | favorable | 0.700: 966/692/0; 0.750: 966/692/0; 0.800: 967/690/1; 0.820: 973/682/3; 0.850: 996/647/15; 0.900: 1127/488/43; 0.940: 1371/220/67; 0.950: 1438/154/66 |
| `C2-best1-vs-C1-best1` | higher | mixed | 0.700: 6/1648/4; 0.750: 6/1648/4; 0.800: 7/1640/11; 0.820: 9/1628/21; 0.850: 22/1575/61; 0.900: 82/1247/329; 0.940: 196/704/758; 0.950: 203/617/838 |
| `D1-vs-C1-best1` | higher | mixed | 0.700: 4/1650/4; 0.750: 4/1650/4; 0.800: 12/1636/10; 0.820: 21/1618/19; 0.850: 50/1556/52; 0.900: 212/1207/239; 0.940: 418/785/455; 0.950: 459/741/458 |
