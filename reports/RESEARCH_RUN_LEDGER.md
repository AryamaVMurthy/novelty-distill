# Research run ledger

This ledger freezes the submitted Turing job graph and completed systems evidence as of
2026-08-02 (Asia/Kolkata). Slurm state is live; job IDs and dependency edges are the durable
record. All active or pending jobs use `codex/implementation` and synchronize through
`.git/novelty-distill-sync.lock` before consuming repository code.

## Production teacher targets

| Stage | Job | Stable artifact or dependency |
|---|---:|---|
| Generation pass 1 | 18031 | `generations/teacher/teacher-1k-v1` |
| Generation pass 2 | 18033 | `afterany:18031`, same resumable directory |
| Quality score pass 1 | 18035 | `afterok:18033` |
| Quality score pass 2 | 18037 | `afterany:18035`, same resumable score directory |
| Score diagnostics | 18123 | `afterok:18037`; strict JSON and Markdown judge diagnostics |
| Cluster and target views | 18124 | `afterok:18037`; cached embeddings; `data/teacher-targets-tomato1k-v1.json` |
| Target provenance/findings gate | 18122 | `afterok:18124`; exact reconstruction, 1,000-ID validation, and descriptive findings |

The second generation and scoring passes are deliberate safety passes. Current preflights validate
all existing shards and exit before model loading when an artifact is already complete.

Generation pass 1 completed successfully in 01:29:52 and pass 2 completed its strict preflight in
00:00:44. The frozen artifact contains exactly 1,000 prompt shards and 8,000 completions under
configuration hash `a6fff8e2deb2278cf2ac5a528df9c44cef791203a4c2073461d6dd44a4e43b79`.
Every prompt has 8/8 distinct completions and all 8,000 completion texts are globally distinct.
The completion-token distribution has mean 355.5487, median 353, and maximum 512. There are 7,985
`stop` finishes and 15 `length` finishes, for a length-stop rate of 0.001875. These are generation
integrity and lexical-uniqueness diagnostics; they do not establish semantic diversity or quality.

The original pending cluster job 18039 was first extended from 01:00:00 to 03:00:00, then replaced
and cancelled before execution by current staged job 18124 so production uses the persistent atomic
embedding cache. The observed seven-condition calibration cluster job processed 56 prompts in
00:04:02; linear scaling projects about 72 minutes for 1,000 prompts, so 18124 retains a three-hour
limit. Training jobs 18097 and 18098 and control evaluation 18113 require the
successful provenance/findings gate 18122, so no model can consume the target artifact before
validation. Gate 18121 was replaced and cancelled while pending because its immutable staged script
predated the deterministic target-view findings report.

## Temporal controls

The original A0/A1 generation jobs write stable prompt shards but cannot finish 1,658 prompts with
16 samples each inside one six-hour allocation at the observed rates. Their replacement graph
preserves those partial shards and deliberately gives the four-GPU DistiLLM smoke job an all-GPU
window before resuming.

| Control | Current pass | Resume passes | Score passes | Final evaluation |
|---|---:|---|---|---:|
| A1 Qwen3-14B | 18078 | 18102 -> 18103 -> 18104 | 18107 -> 18108 | submitted by controller 18125 |
| A0 Qwen3-4B | 18084 | 18105 -> 18106 | 18109 -> 18110 | 18113 -> 18119 |

Jobs 18102 and 18105 require both `afterany` on the current partial generation and `afterok:18092`.
Later generation and score passes use `afterany` and the same stable output IDs. A0 evaluation
18113 requires final A1 score 18108, final A0 score 18110, and target gate 18122; 18119 is its cached
resume pass. Obsolete single-pass
score/evaluation/controller jobs 18080, 18088, 18090, and 18099 were cancelled before execution.

## Training and evaluation matrix

| Work | Job | Contract |
|---|---:|---|
| DistiLLM deployability smoke | 18092 | four GPUs; must produce a loadable full checkpoint after real optimizer updates |
| Primary TRL/OPSD/GEM matrix | 18097 | indices 0-11 and 13-18, at most two concurrent tasks, after target gate 18122 and DistiLLM smoke 18092 |
| TRL/OPSD bounded resumes | 18117 | indices 0-4, 6-11, and 13-18; `afterany:18097`; 25-step checkpoints |
| C3 DistiLLM | 18098 | index 12, four GPUs; after target gate 18122 and smoke 18092 |
| Evaluation fan-out controller | 18125 | waits for B4 task 18097_5, 18117, 18098, A0/A1, and validated targets 18122 |

Job 18097 deliberately waits for 18092 as well as the target gate. Otherwise a one-GPU training
task could occupy the only GPU released by teacher scoring/clustering and starve the four-GPU
DistiLLM proof when the current temporal-control jobs release the remaining devices.

Controller 18125 replaces pending controller 18120, whose immutable export still named cancelled
cluster job 18039. Its internal target dependency is validation gate 18122. It submits A1 and
historical A3 controls plus 19 trained model chains. Each trained
model chain has three resumable generation passes, two resumable judge passes, two cached joint
evaluation passes, and strict checkpoint preflight. The final CPU analysis requires all aligned evaluation artifacts,
runs the frozen prompt-paired contrasts with per-metric Holm correction, and reports per-prompt
favorable/tied/unfavorable directions over every declared clustering threshold.

## UltraFeedback systems pilot

Array 18043 provides an independent, pinned 1,000-row systems check before TOMATO targets are
available. Completed tasks 0-3 produced deployable 69,782,384-byte LoRA adapters:

| Baseline | Steps | Epoch fraction | First loss | Last loss | Last gradient norm | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 125 | 1.00 | 1.8238 | 0.8383 | 0.3780 | 00:03:00 |
| B2a | 125 | 1.00 | 2.4363 | 1.3320 | 0.5553 | 00:02:58 |
| B2b | 125 | 1.00 | 1.8238 | 0.8385 | 0.3666 | 00:02:59 |
| B3 | 125 | 0.25 | 2.0011 | 1.3974 | 0.7914 | 00:03:01 |

B3's 0.25 epoch is expected: four targets create 4,000 available rows while the frozen gate uses
1,000 optimizer example exposures. UltraFeedback's canonical `human_target`, `best1`, and `mode1`
are all the same preferred completion by construction. Therefore B1 and B2b are target-equivalent
in this pilot; their nearly identical loss traces are a pipeline consistency finding, not a valid
human-versus-best-target comparison. B2a uses the deterministic random completion and B3 uses all
four unique official completions. D1 task 18043_4 remains the expensive on-policy pilot and must
finish with a deployable adapter before that backend is considered research-ready.

## Interpretation gates

- The 1,000-row stage is a systems and first-effect gate, not final evidence.
- Four-target methods are budget-matched samples from a 4,000-row pool at this stage; promoted
  claims require a full-pool exposure sensitivity and repeated single-target controls.
- The fixed judge and embedding clusterer are reproducible proxies, not human-calibrated scientific
  novelty labels.
- No semantic-diversity claim is promotable from one threshold; all seven thresholds and prompt
  direction counts must be inspected.
- C3 has no verified optimizer-state resume path in the pinned official implementation. Its
  four-GPU smoke must establish that the full 250-step run fits the six-hour cluster limit before
  the C3 dependency can be considered robust.
