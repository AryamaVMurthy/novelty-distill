# D2 K=4 extension run audit

Analysis producer commit: `453f075f4ef697e086504b86432c655eb15813c5`.

## Training artifact

- Run: `D2-tomato1k-seed17`.
- Method: on-policy reverse-KL distillation with student trajectories,
  `beta=1`, `lambda=1`.
- Student: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Data: 1,000 TOMATO training examples, 1,000 example exposures, seed 17.
- Completion: 125/125 steps in 13,638.43 seconds; final logged train loss
  1.060318.
- Adapter identity:
  `sha256:56904eba6ecb94626bd6d9cb10f1afda48ce5a7571b6f3a6181f4db662359c46`.
- The canonical 15-method subset audit independently verified the adapter,
  dataset, ordered IDs, input bytes, base revision, seed, and exposure budget.

## Held-out evaluation completeness

- Prompts: 1,658 held-out TOMATO prompts.
- Generation: 1,658 prompt files x 4 candidates = 6,632 D2 hypotheses.
- Judge scoring: 1,658 prompt files x 4 records = 6,632 complete records.
- Generation config hash:
  `1b9e9cdff0df3636e611748a32d6d80fb0c4eeaec6b5a5489a56d0351ed6561d`.
- Input hash:
  `0058f63d0e4004d8a50cec5b7a92cfc27171aa3acbddffd3d8556a4dba2a5925`.
- D2 score-tree hash:
  `9bf996c106f3139b90fbb31d8830928d4fbab5696e5104bc93f68343c29f47b2`.
- Evaluation JSON hash:
  `6096d196a9d2361b65e8b619c64cac71c83e30af45e11f18169d9cc65eef7c76`.

Generation task `18730_2` and recovery-array tasks `18740_0`, `18740_1`, and
`18740_3` completed in about 41 minutes each. Gate `18743` validated the exact
prompt count, candidate count, generation configuration, and adapter identity.

The first three array tasks failed immediately because concurrent jobs raced
while creating the node05 repository clone; no generation output was lost or
accepted from a failed task. Only the three missing shards were resubmitted.
This was an infrastructure recovery and did not change an experimental input.

## Judge and embedding provenance

- Frozen judge: `Qwen/Qwen3-32B-FP8`, revision
  `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.
- Shared teacher-score hash:
  `9227bb39afaecace106ea8f3a0b3ed87706145c7a3831f9bed0c11c22463de4b`.
- Embedding model: `Qwen/Qwen3-Embedding-4B`, revision
  `5cf2132abc99cad020ac570b19d031efec650f2b`.
- Training-reference targets: 8,000, hash
  `eea6b6e35f5d5e181740f656a5ec49a29ed40daa21ba3297ab26dc0ffc2fbd9f`.
- Teacher samples: first four from the validated K=16 source. D2 samples:
  four newly generated candidates. All comparisons are K=4 matched.

The first scoring pass (`18744`) was configured for concurrency eight per L40S
server, and two tasks exhausted transient activation memory after writing 100
valid prompt files. The pass was stopped and resumed idempotently as array
`18753` with concurrency two. Judge model, revision, prompts, rubric,
temperature, and outputs were unchanged. All four recovery tasks exited 0,
and gate `18754` validated 1,658 complete prompt files with zero pending.

Embedding evaluation `18755` completed in 00:05:10. It reused 14,632 cached
representations and embedded all 6,632 new D2 hypotheses. Analysis `18756`
completed in 15 seconds and emitted 11,606 primary rows (7 methods x 1,658)
and 92,848 threshold rows (7 methods x 1,658 x 8 thresholds). The final Turing
queue audit was empty.

## Statistical contract

The extension declares three paired treatment-minus-reference contrasts:
`D2-vs-D1`, `D2-vs-C2-best1`, and `D2-vs-C1-best1`. Confidence intervals and
sign-flip p-values use 10,000 Monte Carlo samples with seed 17. Holm correction
is applied across these three contrasts separately within each primary metric.
The original compact study's five-contrast family is untouched.

Large-output hashes:

- `paired-contrasts.json`:
  `0dd509f439e060605fa0bc4ba5b5e456a900aa5f915227b61403ed4023f40687`
- `prompt-metrics.jsonl`:
  `5e82333a46c29049804e16fa55791f806cabd91aa0c0697b5363c4a785b821a5`
- `threshold-prompt-metrics.jsonl`:
  `26507ba44f44d43549613d36890b085d1b14b812ee1407479410c85bdf246020`

## Interpretation boundary

These are frozen automatic-judge and embedding outcomes. They measure
operational quality, fidelity to teacher modes, and quality-gated semantic
breadth; they are not human validation of scientific novelty. ClusterJSD and
threshold-dependent metrics are only comparable under this shared K=4,
clustering, and threshold protocol.
