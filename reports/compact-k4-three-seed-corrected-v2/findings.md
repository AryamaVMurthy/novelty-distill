# Compact K=4 three-seed checkpoint analysis

Prompt-paired intervals within each run characterize test-prompt uncertainty, while the across-seed summary uses the independently trained checkpoint seed as the replication unit. With only three checkpoint seeds, the across-seed t interval is low-powered and descriptive. Qwen feasibility and soundness are operational judge measurements, not human ground truth or measurements of global scientific novelty. Semantic threshold metrics remain quarantined until blinded human equivalence calibration is completed.

## student_feasibility_mean

| Contrast | Mean delta | Seed SD | 95% t CI | Seed deltas |
|---|---:|---:|---:|---|
| B2a-vs-A0 | -0.8323 | 0.0217 | [-0.8862, -0.7785] | 17: -0.8200, 29: -0.8574, 43: -0.8197 |
| B2b-vs-B2a | 0.0216 | 0.0250 | [-0.0406, 0.0837] | 17: 0.0008, 29: 0.0493, 43: 0.0146 |
| C1-best1-vs-B2b | 0.8379 | 0.0061 | [0.8228, 0.8529] | 17: 0.8402, 29: 0.8310, 43: 0.8424 |
| C2-best1-vs-C1-best1 | -0.0059 | 0.0100 | [-0.0308, 0.0189] | 17: 0.0014, 29: -0.0018, 43: -0.0173 |
| D1-vs-C1-best1 | -0.0056 | 0.0030 | [-0.0131, 0.0018] | 17: -0.0050, 29: -0.0089, 43: -0.0030 |
| D2-vs-C2-best1 | -0.0104 | 0.0018 | [-0.0150, -0.0058] | 17: -0.0092, 29: -0.0125, 43: -0.0095 |
| D2-vs-D1 | -0.0107 | 0.0114 | [-0.0391, 0.0177] | 17: -0.0029, 29: -0.0054, 43: -0.0238 |

## student_soundness_mean

| Contrast | Mean delta | Seed SD | 95% t CI | Seed deltas |
|---|---:|---:|---:|---|
| B2a-vs-A0 | -1.4364 | 0.0135 | [-1.4699, -1.4028] | 17: -1.4371, 29: -1.4495, 43: -1.4225 |
| B2b-vs-B2a | 0.0235 | 0.0183 | [-0.0219, 0.0690] | 17: 0.0107, 29: 0.0445, 43: 0.0154 |
| C1-best1-vs-B2b | 1.4729 | 0.0101 | [1.4478, 1.4979] | 17: 1.4813, 29: 1.4617, 43: 1.4756 |
| C2-best1-vs-C1-best1 | -0.0079 | 0.0091 | [-0.0306, 0.0147] | 17: -0.0017, 29: -0.0038, 43: -0.0184 |
| D1-vs-C1-best1 | 0.0031 | 0.0019 | [-0.0017, 0.0079] | 17: 0.0009, 29: 0.0039, 43: 0.0045 |
| D2-vs-C2-best1 | -0.0118 | 0.0098 | [-0.0361, 0.0126] | 17: -0.0178, 29: -0.0170, 43: -0.0005 |
| D2-vs-D1 | -0.0228 | 0.0022 | [-0.0284, -0.0173] | 17: -0.0204, 29: -0.0247, 43: -0.0234 |
