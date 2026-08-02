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
| Cluster and target views | 18131 | cached complete-linkage 0.94 rebuild; `data/teacher-targets-tomato1k-v1.json` |
| Target provenance/findings gate | 18132 | exact reconstruction, 1,000-ID validation, and descriptive findings |

The second generation and scoring passes are deliberate safety passes. Current preflights validate
all existing shards and exit before model loading when an artifact is already complete.

Generation pass 1 completed successfully in 01:29:52 and pass 2 completed its strict preflight in
00:00:44. The frozen artifact contains exactly 1,000 prompt shards and 8,000 completions under
configuration hash `a6fff8e2deb2278cf2ac5a528df9c44cef791203a4c2073461d6dd44a4e43b79`.
The filename-NUL-content tree serialization is 16,860,476 bytes with SHA-256
`151b5b79cb65126a84abcbc8a92276215a0f1a2b05b8d3aede3f494a1afb8d32`.
Every prompt has 8/8 distinct completions and all 8,000 completion texts are globally distinct.
The completion-token distribution has mean 355.5487, median 353, and maximum 512. There are 7,985
`stop` finishes and 15 `length` finishes, for a length-stop rate of 0.001875. These are generation
integrity and lexical-uniqueness diagnostics; they do not establish semantic diversity or quality.

Quality score pass 1 completed successfully in 01:29:07 and pass 2 completed its strict preflight
in 00:00:49. The frozen score tree contains exactly 1,000 schema-v2 shards and 8,000 records, all
with unique request IDs, under tree SHA-256
`3c2986fe0a6ca2a985737a177f2e6fc3ed3ac3e669c553f0dfa7a5b437989e7f`.
Diagnostics job 18123 completed in 00:00:02. Aggregate quality is 0.827656 (population SD 0.104174),
the mean within-prompt range is 0.19215, and the aggregate ceiling rate is 0.0585. Instruction
compliance is saturated at a 0.9925 rating-5 rate, whereas feasibility and soundness retain
headroom with means 3.733625 and 3.71425. Global and prompt-centered quality--length correlations
are 0.078351 and 0.111392. These results establish judge resolution and expose dimension-specific
ceilings; they do not validate the rubric against human scientific judgments. Full frozen values
are recorded in `reports/TEACHER_SCORE_FINDINGS.md`.

The original pending cluster job 18039 was replaced before execution by cached job 18124. That job
completed the first production clustering in 00:11:49 and gate 18122 exposed a teacher-only
calibration failure: the original 0.82 threshold produced only 1.001 modes per prompt. Before any
student output was inspected, an intermediate connected-component 0.95 target was built by job
18127 and validated by gates 18128/18129. A deterministic qualitative audit then found obvious
paraphrase splitting and the bridge-chaining transition. Complete-linkage 0.94 was selected from a
teacher-only whole-corpus linkage audit because it enforces the threshold for every pair within a
mode. Final cached rebuild 18131 completed in 00:01:52 and final gate 18132 in 00:00:03. Superseded
jobs remain in the ledger as transparent calibration history; the current release gate is 18132.

The final target has 3.798 instructed modes per prompt (median 4), and 54.2% of prompts have at
least four measured modes. `diverse4` contains 3.146 modes per prompt on average. Its quality mean
is 0.866088 versus 0.9166 for best-1, exposing the intended selection trade-off. Mode-1 equals
best-1 on 603 prompts, so their direct contrast is informative on the remaining 397 prompts. Raw
and instructed counts still differ materially; the complete eight-threshold curve and expert-review gate
remain mandatory. Exact hashes and the full distribution are in
`reports/TEACHER_TARGET_FINDINGS.md`.

Before any student metric was computed, evaluation commit `40e9244` closed an asymmetric
assignment loophole: a student now joins a frozen teacher mode only when its minimum cosine to
every member reaches the threshold. Choosing the qualifying mode by its best complete-link score
keeps the 0.94 boundary consistent between teacher partitioning and student admission; otherwise a
student could match one member while contradicting another. Unmatched students remain
complete-linkage clustered as separately named student modes.

Training jobs 18092, 18097, and 18098 and control evaluation 18113 directly require successful
gate 18132, so no target consumer can run on a superseded artifact.

## Temporal controls

The original A0/A1 generation jobs write stable prompt shards but cannot finish 1,658 prompts with
16 samples each inside one six-hour allocation at the observed rates. Their replacement graph
preserves those partial shards and deliberately gives the four-GPU DistiLLM smoke and C3 production
jobs a contiguous all-GPU window before resuming.

| Control | Current pass | Resume passes | Score passes | Final evaluation |
|---|---:|---|---|---:|
| A1 Qwen3-14B | 18078 | 18102 -> 18103 -> 18104 | 18107 -> 18108 | submitted by controller 18133 |
| A0 Qwen3-4B | 18084 | 18105 -> 18106 | 18109 -> 18110 | 18113 -> 18119 |

Jobs 18102 and 18105 require `afterany` on both the current partial generation and C3 job 18098.
Later generation and score passes use `afterany` and the same stable output IDs. A0 evaluation
18113 requires final A1 score 18108, final A0 score 18110, and target gate 18132; 18119 is its cached
resume pass. Obsolete single-pass
score/evaluation/controller jobs 18080, 18088, 18090, and 18099 were cancelled before execution.

## Training and evaluation matrix

| Work | Job | Contract |
|---|---:|---|
| DistiLLM deployability smoke | 18092 | four GPUs; after target gate 18132; must produce a loadable full checkpoint after real updates |
| DistiLLM SGLang load gate | 18141 | after smoke 18092; C3 production cannot start until the full checkpoint generates successfully |
| Primary TRL/OPSD/GEM matrix | 18097 | indices 0-11 and 13-18, at most two concurrent tasks, after target gate 18132 and `afterany:18098` |
| TRL/OPSD bounded resumes | 18117 | indices 0-4, 6-11, and 13-18; `afterany:18097`; 25-step checkpoints |
| C3 DistiLLM | 18098 | index 12, four GPUs, 12-hour bound; after target gate 18132 and smoke 18092 |
| Evaluation fan-out controller | 18133 | waits for B4 task 18097_5, 18117, 18098, A0/A1, and names target gate 18132 |

Pre-production context gates 18134--18138 ran on the longest tokenizer-audited TOMATO record.
Task-faithful OPSD at 3,072 tokens passed in 18134. GKD 3,072 failed closed on memory in 18135, and
the first 2,048 retry 18136 exposed the upstream prompt-dropping truncation edge. Commit `141f31f`
introduced a prompt-preserving ChatML adapter with the identical tensor contract; static retry
18137 and on-policy smoke 18138 then completed with real optimizer steps and deployable adapters.
Production contexts are frozen at SFT 3,072, GKD 2,048, OPSD 3,072, GEM 1,024, and DistiLLM 896.
The GKD metadata records completion truncation counts/tokens and the explicit non-thinking prompt
mode for every run.
Final on-policy prompt-contract gate 18139 completed in 00:00:50 at commit `67c9a48`, recording
`student_thinking: false`, zero truncation, finite loss/gradient, and a deployable adapter.
The subsequent decoding-parity gate 18140 completed in 00:00:50 at commit `6eb1c5a`, after a
source audit found that official GKD otherwise forced `top_k=0` and inherited Qwen's
`top_p=0.95`. The successful run records the frozen teacher/evaluation controls
`temperature=0.7`, `top_p=0.8`, `top_k=20`, and `min_p=0.0`, together with non-thinking mode,
zero truncation, loss 0.6325, gradient norm 1.9707, and a structurally valid 504-tensor adapter.

A pre-execution audit of the pinned DistiLLM loop found that its distributed sampler drops rows
to a complete world-size batch and its original one-epoch launch would yield only 237 steps from
950 train rows, never reaching the step-250 save gate. Production C3 now uses 960 train and 40
validation rows, exactly 240 four-example steps per epoch, and computes two epochs to stop at step
250. This preserves the common 1,000 optimizer-example exposure budget and guarantees that the
official step-250 checkpoint branch is reachable.
The same audit found that the paper's adaptive scheduler starts at replay probability zero and
raises it over ten validation stages when held-out loss worsens. The previous `max_steps + 1`
validation interval would never call that scheduler. C3 now runs the paper-aligned 25-step
validation cadence and fails closed unless the official log contains every optimizer step and a
consistent validation-loss/threshold trajectory. This preserves an adaptive-replay test rather
than silently relabeling static skew-KL as DistiLLM.
Finally, the pinned Dolly preprocessor was found to emit a legacy Qwen string without Qwen3's
`user`/`assistant` roles. C3 now uses a task adapter that renders Qwen3's official non-thinking
chat template and delegates indexed-file construction to DistiLLM's official mmap builder. The
adapter preserves the prompt, records completion-tail truncation, and rejects content containing
the upstream loader's hard-coded separator token ID 65535 rather than risking a false prompt
boundary.
Exact pinned-tokenizer audit 18143 then passed all 1,000 task-adapter rows: zero separator
collisions, maximum prompt length 453 under the 480-token cap, and one best-1 completion truncated
by 19 tokens at the 896-token full-chat cap. Job 18142 was a shell-wrapper failure before the audit
started; it produced no data or model artifact.
The first small regression 18144 failed before model loading because its obsolete 192-token cap
correctly rejected the 223-token Qwen3 prompt. After raising only this systems-smoke cap, retry
18145 completed in 00:00:51: one official optimizer step at loss 0.8328, peak observed GPU memory
46,472 MiB, exactly two normalized separators, a complete step log, and a 3,441,191,930-byte full
student weight file. Its two 512-token pilot completion truncations are smoke-only; the production
896-token rate remains the exact 1/1,000, 19-token audit above. Per-invocation byte offsets now
isolate official log audits while preserving all earlier retry history. Regression 18146 reran the
same optimizer step at commit `9c17194` and confirmed that the current invocation contains exactly
one validation check and one logged training step despite the preserved historical log.
Checkpoint-serving jobs 18147 and 18148 then isolated a device-specific deterministic-inference
failure: the batch-invariant Triton matmul requests 106,496 bytes of shared memory, above node01's
101,376-byte limit, and the failure persists with CUDA graphs disabled. Standard eager job 18149
passed weight loading but exposed TVM-FFI's independent home-cache default; commit `26f246a` now
sets `TVM_FFI_CACHE_DIR` to project scratch and verifies it is writable. Job 18150 contained a
mistyped, nonexistent checkpoint root and failed before loading. Corrected retry 18151 completed in
00:01:06, loaded the 3,441,191,930-byte Qwen3-1.7B checkpoint, compiled scratch-cached JIT kernels,
and produced two non-empty, non-thinking 64-token generations from a 223-token TOMATO prompt. The
four-GPU smoke 18092 therefore returned to `PENDING (Resources)` with no unsatisfied dependency.

Job 18097 and the first A0/A1 resume passes wait until C3 job 18098 terminates. This reserves the
all-GPU sequence 18092 -> 18098 before one-GPU work can occupy a released device. Their `afterany`
edges release the other baselines and controls even if C3 fails, while controller 18133 separately
requires C3 success before evaluation fan-out.

Controller 18133 replaces pending controllers 18120, 18125, 18130; their immutable exports named
superseded target gates. Its internal target dependency is validation gate 18132. It submits A1 and
historical A3 controls plus 19 trained model chains. Each trained
model chain has three resumable generation passes, two resumable judge passes, two cached anchored
evaluation passes, and strict checkpoint preflight. The final CPU analysis requires all aligned evaluation artifacts,
runs the frozen prompt-paired contrasts with per-metric Holm correction, and reports per-prompt
favorable/tied/unfavorable directions over every declared clustering threshold.

The registry contains 24 planned entries: four non-training controls, nineteen executable training
variants, and E1. E1 is explicitly `fail_closed` because the pinned official OPSD implementation
does not expose static-trajectory off-policy training. It is neither missing nor silently skipped,
and it cannot enter the executable matrix without a reviewed upstream implementation.

## UltraFeedback systems pilot

Array 18043 provides an independent, pinned 1,000-row systems check before TOMATO targets are
available. Completed tasks 0-4 produced deployable 69,782,384-byte LoRA adapters:

| Baseline | Steps | Epoch fraction | First loss | Last loss | Last gradient norm | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 125 | 1.00 | 1.8238 | 0.8383 | 0.3780 | 00:03:00 |
| B2a | 125 | 1.00 | 2.4363 | 1.3320 | 0.5553 | 00:02:58 |
| B2b | 125 | 1.00 | 1.8238 | 0.8385 | 0.3666 | 00:02:59 |
| B3 | 125 | 0.25 | 2.0011 | 1.3974 | 0.7914 | 00:03:01 |
| D1 | 125 | 1.00 | 5.8326 | 2.9114 | 2.6754 | 04:58:09 |

B3's 0.25 epoch is expected: four targets create 4,000 available rows while the frozen gate uses
1,000 optimizer example exposures. UltraFeedback's canonical `human_target`, `best1`, and `mode1`
are all the same preferred completion by construction. Therefore B1 and B2b are target-equivalent
in this pilot; their nearly identical loss traces are a pipeline consistency finding, not a valid
human-versus-best-target comparison. B2a uses the deterministic random completion and B3 uses all
four unique official completions. D1 task 18043_4 completed all 125 on-policy steps with mean
training loss 3.5920, peak observed GPU memory 23,970 MiB, and a readable 69,782,384-byte,
392-tensor adapter. It used the earlier upstream UltraFeedback prompt/sampler and is systems
evidence only; TOMATO production uses the separately gated task-faithful, non-thinking,
frozen-decoding contract.

## Interpretation gates

- The 1,000-row stage is a systems and first-effect gate, not final evidence.
- Four-target methods are budget-matched samples from a 4,000-row pool at this stage; promoted
  claims require a full-pool exposure sensitivity and repeated single-target controls.
- The fixed judge and embedding clusterer are reproducible proxies, not human-calibrated scientific
  novelty labels.
- No semantic-diversity claim is promotable from one threshold; all eight thresholds and prompt
  direction counts must be inspected.
- C3 has no verified optimizer-state resume path in the pinned official implementation. Its
  production allocation is therefore 12 hours instead of the default six-hour research-job bound.
  This is within the observed `u22` unlimited partition limit and the `high` QOS seven-day maximum;
  the four-GPU smoke still must prove a real update and deployable checkpoint before C3 releases.
- The pinned-tokenizer corpus audit found no SFT/OPSD overflow at 3,072 and no prompt or on-policy
  overflow at GKD 2,048. Three of 1,000 historical human targets still require prompt-preserving
  completion-tail truncation at the hardware-safe GKD limit; this 0.3% rate and token count remain
  mandatory diagnostics. DistiLLM's 896-token full-chat proxy exceeds its cap for 1/1,000 records
  by 19 tokens, while all prompts fit its separate 480-token prompt cap.
- GKD on-policy sampling is explicitly overwritten on the official trainer after construction and
  recorded in run metadata. Model defaults are not accepted as implicit experimental controls.
