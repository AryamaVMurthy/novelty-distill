# Random-1 sequence-KD extension (B2a), corrected K=4

## Result

Random-target sequence KD (`B2a`) fails almost identically to best-of-eight
sequence KD (`B2b`) under the frozen Qwen judge. This rules out best-of-eight
target selection as the main explanation for the seed-17 hard-KD collapse.
The more plausible baseline conclusion is that hard single-output imitation,
at this 1,000-example/one-epoch setting, is the failure mode; token-distribution
forward-KL training (`C1-best1`) avoids it.

| Method | Feasibility /5 | Soundness /5 | Completion tokens | Strict yield at 0.94* |
|---|---:|---:|---:|---:|
| A0 base | 4.2176 | 4.6025 | 329.0 | 1.8776 |
| B2a random-1 SeqKD | 3.3976 | 3.1654 | 212.7 | 0.4530 |
| B2b best-1 SeqKD | 3.3984 | 3.1761 | 212.4 | 0.4602 |
| C1 off-policy forward KL | 4.2385 | 4.6574 | 347.2 | 2.2238 |

`*` Semantic yield is a threshold-curve diagnostic only. The 0.94 boundary is
not calibrated against human equivalence labels and cannot support primary
inference.

Primary paired results over 1,658 prompts:

- B2a versus A0: feasibility -0.8200, 95% CI [-0.8382, -0.8016];
  soundness -1.4371, 95% CI [-1.4573, -1.4165].
- B2b versus B2a: feasibility +0.00075, 95% CI [-0.01327, +0.01448];
  soundness +0.01071, 95% CI [-0.00242, +0.02397]. Neither difference is
  distinguishable after the declared Holm correction.
- C1 versus B2a: feasibility +0.8409, 95% CI [+0.8231, +0.8586];
  soundness +1.4920, 95% CI [+1.4717, +1.5122].

The target construction is genuinely different: only 110/1,000 random-1
targets exactly equal best-1, and the random target index is roughly uniform
over the eight teacher candidates. Nonetheless, paired held-out Qwen scores
for B2a and B2b are extremely similar. Exact output text matches are rare
(0.09%), so this is behavioral convergence rather than duplicate output files.

These are one-checkpoint, Qwen-judge operational findings—not global novelty,
human quality, or across-training-seed evidence. The frozen three-seed
replication is now running and is the deciding result.

## Files

- [`findings.md`](findings.md): complete predeclared paired analysis and every
  semantic threshold direction table.
- [`paired-contrasts.json`](paired-contrasts.json): machine-readable estimates.
- [`baseline-summary.csv`](baseline-summary.csv): compact method-level table.
- [`quality.svg`](quality.svg) and [`breadth.svg`](breadth.svg): frozen plots.
- [`RUN_AUDIT.md`](RUN_AUDIT.md): execution, completeness, and provenance.
