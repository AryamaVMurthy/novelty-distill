# Research run ledger

This ledger freezes the submitted Turing job graph and completed systems evidence as of
2026-08-02 (Asia/Kolkata). Slurm state is live; job IDs and dependency edges are the durable
record. All newly staged jobs use `codex/implementation` and synchronize through
`.git/novelty-distill-sync.lock` before consuming repository code. Primary array 18097 was spooled
before that lock was added; it runs one task at a time while controls occupy the other GPUs, and
locked resume array 18195 covers any incomplete artifact.

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

Training jobs 18092, 18097, and 18098 directly required successful gate 18132. The replacement
evaluation graph instead culminates in final target replay 18209, so no final metric consumer can
run on a superseded artifact.

## Temporal controls

The original A0/A1 generation jobs write stable prompt shards but cannot finish 1,658 prompts with
16 samples each inside one six-hour allocation at the observed rates. Their replacement graph
preserves those partial shards and deliberately gives the four-GPU DistiLLM smoke and C3 production
jobs a contiguous all-GPU window before resuming.

| Control | Current pass | Resume passes | Score passes | Final evaluation |
|---|---:|---|---|---:|
| A1 Qwen3-14B | 18078 | 18162 -> 18198 -> 18199 | 18201 -> 18202 | submitted by controller 18210 |
| A0 Qwen3-4B | 18084 | 18165 -> 18200 | 18203 -> 18204 | 18207 -> 18208 |

The first replacement wrappers 18102--18106 used `/bin/sh` despite containing Bash's `pipefail`;
all five failed in at most one second before repository synchronization, model loading, or data
work. Direct Bash-script replacements 18162--18166 preserved exactly the same output IDs and
completed shards. A pre-execution batch-script audit found the same `/bin/sh` defect in queued score
jobs 18107--18110 and evaluation job 18113; all were cancelled with zero runtime, along with its
otherwise-correct downstream pass 18119. Direct-script replacements 18178--18181 score the same
stable A1/A0 namespaces, and 18184--18185 evaluate A0 against A1 after both final score passes.
Before those pending continuations ran, a second audit found that repository synchronization was
locked but shared virtual-environment mutation was not. Running passes 18162/18165 remain valid;
lock-bearing continuations 18198--18200, score passes 18201--18204, fresh target gate 18206, and
A0 evaluation passes 18207--18208 replace the still-zero-runtime jobs 18163--18166, 18178--18181,
and 18184--18185. Controller 18210 now waits for final score 18202 and control evaluation 18208.
The older one-prompt base audit 18177 now waits `afterany:18208`; its downstream C3 rerun 18186 and
B4 proof 18191 remain serial, isolating those already-spooled scripts from all environment setup.
Obsolete single-pass
score/evaluation/controller jobs 18080, 18088, 18090, and 18099 were cancelled before execution.
The active passes 18078 and 18084 were ended once their atomic shards reached 523/1,658 A1 prompts
and 983/1,658 A0 prompts, respectively, so the requested four-GPU gate could start immediately.
Their resume jobs retain the same output IDs and begin at the first missing prompt after C3.

## Training and evaluation matrix

| Work | Job | Contract |
|---|---:|---|
| DistiLLM deployability smoke | 18092 | four GPUs; after target gate 18132; must produce a loadable full checkpoint after real updates |
| DistiLLM SGLang load gate | 18141 | after smoke 18092; C3 production cannot start until the full checkpoint generates successfully |
| Primary TRL/OPSD/GEM matrix | 18097 | indices 0-11 and 13-18, at most two concurrent tasks, after target gate 18132 and `afterany:18098` |
| TRL/OPSD bounded resumes | 18195 | indices 0-4, 6-11, and 13-18; `afterany:18097`; 25-step checkpoints |
| C3 DistiLLM | 18098 | index 12, four GPUs, 12-hour bound; after target gate 18132 and smoke 18092 |
| Final target replay | 18209 | waits for 18195, final A1 score 18202, and final A0 evaluation 18208 |
| Evaluation fan-out controller | 18210 | waits for final target replay 18209 and B4 deploy gate 18191; names the fresh target gate for every submitted evaluation chain |

Completed TOMATO-1k production runs currently have the following measured systems costs. Training
losses are intentionally omitted from this cross-backend table because CE, GEM, forward/reverse
KL, and skew KL have different numerical scales.

| Baseline | Objective/view | Available rows | Exposures | Train runtime (s) | Peak GPU MiB |
|---|---|---:|---:|---:|---:|
| B2a | CE / random-1 | 1,000 | 1,000 | 245.9 | 10,902 |
| B2b | CE / best-1 | 1,000 | 1,000 | 245.1 | 10,982 |
| B2c | CE / mode-1 | 1,000 | 1,000 | 251.0 | 10,924 |
| B4 | GEM / diverse-4 | 4,000 | 1,000 | 340.9 | 39,626 |
| C1-human | forward KL / human | 1,000 | 1,000 | 617.7 | 45,662 |
| C1-best1 | forward KL / best-1 | 1,000 | 1,000 | 487.2 | 40,298 |
| C1-diverse4 | forward KL / diverse-4 | 4,000 | 1,000 | 485.8 | 40,498 |
| C2-human | reverse KL / human | 1,000 | 1,000 | 617.0 | 43,020 |
| C2-best1 | reverse KL / best-1 | 1,000 | 1,000 | 489.9 | 40,024 |
| C2-diverse4 | reverse KL / diverse-4 | 4,000 | 1,000 | 487.7 | 40,078 |
| C3 | skew KL / best-1 | 1,000 | 1,000 | 1,846.0 | 48,502 |

All nine completed LoRA artifacts above B4/C3 open as 132,187,888-byte rank-16 adapters with
504 non-empty tensors. B4 is a structurally valid 8,044,981,992-byte, 398-tensor native-BF16 full
model; C3's full-checkpoint and BF16-cast serving evidence is recorded below. The three long human
targets in C1/C2 account for 1,446 completion-tail tokens and materially increase runtime and peak
memory relative to the teacher-target views; this context-cost difference must remain visible in
method comparisons.

The checkpoint-125 trainer histories provide a separate convergence diagnostic. Values below are
25-step window means, so they are comparable only within an objective and are not downstream
quality metrics.

| Baseline | First-window loss | Final-window loss | Change | Final-window gradient norm |
|---|---:|---:|---:|---:|
| B2a | 1.1235 | 0.6400 | -43.0% | 0.4247 |
| B2b | 1.1172 | 0.6485 | -42.0% | 0.4165 |
| B2c | 1.1046 | 0.6466 | -41.5% | 0.4169 |
| C1-human | 3.4759 | 2.5733 | -26.0% | 1.9729 |
| C1-best1 | 3.8937 | 2.5084 | -35.6% | 2.4118 |
| C1-diverse4 | 3.9038 | 2.4845 | -36.4% | 2.3711 |
| C2-human | 3.4503 | 2.9953 | -13.2% | 3.7836 |
| C2-best1 | 3.4373 | 2.8166 | -18.1% | 3.6726 |
| C2-diverse4 | 3.4013 | 2.7882 | -18.0% | 3.7093 |

All windows are finite and show net loss reduction. The three SeqKD target views end within 1.4%
of one another. Reverse-KL retains materially larger late-window gradient norms than forward-KL
on the same model and exposure budget; this is an optimization finding only, and the frozen paired
evaluation must determine whether it corresponds to useful behavioral differences.

On-policy D1 started as array task 18097_13 (concrete job 18193) and completed its first two real
steps in 265.9 and 278.8 seconds, with both 4B student and 14B teacher resident at approximately
38.6 GiB GPU memory. This projects beyond one six-hour allocation, but the run saves every 25
steps and the bounded resume array consumes its latest complete checkpoint. D2 initially remained
resource-pending even though a fourth GPU was free: D1 plus the two controls reserved 256 GiB of
the node's 386,630 MiB, leaving slightly less than D2's default 128 GiB request. Its pending-only
reservation was conservatively reduced to 116 GiB (the identical live D1 workload's measured host
RSS was about 1.6 GiB), after which concrete job 18194 started without restarting any active work.
The node therefore has four useful GPU lanes rather than one scheduler-idle device. Pending resume
array 18195 now uses the same 116 GiB reservation, allowing its two-task throttle to coexist with
both 64 GiB controls under the node's measured RAM ceiling; this changes only Slurm reservation
accounting, not any batch, optimizer, context, or checkpoint setting.

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
After preserving the completed A0/A1 shards and ending their resumable active passes, 18092 started
immediately and completed in 00:01:11. Its official four-rank update recorded train loss 0.9424,
initial dev loss 5.3555, no truncation among five rows, peak observed GPU memory 48,174 MiB, and an
8,044,991,278-byte step-1 Qwen3-4B checkpoint. Gate 18141 loaded those weights but failed in the
same deterministic Triton kernel: DistiLLM's DeepSpeed recipe writes FP16 whereas the base
Qwen3-4B is BF16. Standard eager replacement 18154 completed in 00:01:34 and generated 16/16
non-empty, non-thinking samples, all with `stop` finish reasons and 320--382 completion tokens.
Production C3 task 18098_12 then started automatically on four GPUs and completed all 250 steps in
00:30:46. It consumed exactly 1,000 optimizer-example exposures, crossed the planned epoch boundary
from step 240 to 241, wrote the 8,044,991,278-byte full checkpoint at step 250, and peaked at
48,502 MiB on a 49,140 MiB device. The official log contains exactly 250 training steps and all 11
planned validation checks. Validation loss moved from 1.45293 initially to 1.11865 at step 250;
the final three training losses were 0.8050, 0.7788, and 0.7991. All 11 adaptive thresholds stayed
at zero because no validation loss crossed the scheduler's initial/reference-loss-plus-0.1 trigger.
Thus the adaptive scheduler was enabled and fully audited, but this particular task/seed realized
a static teacher skew-KL trajectory rather than activating student replay. Context auditing found
one truncated completion out of 1,000 (19 tail tokens), maximum prompt length 453, maximum
untruncated chat length 915, and exactly 1,000 normalized separators.

Commit `13fe3df` makes dtype an explicit validated serving input and configures BF16 only for C3's
full-checkpoint generation and official evaluation. Confirmation 18156 unexpectedly launched with
`dtype=auto` and therefore reproduced the known FP16 shared-memory failure; it did not test the
BF16 hypothesis. Commit `85757ae` logs the effective mode before server launch. Replacement 18167
then completed in 00:00:48 with `dtype=bfloat16`, deterministic inference, and CUDA graphs enabled;
the server explicitly cast the FP16 checkpoint, allocated a BF16 KV cache, captured every planned
graph batch, and generated successfully. Production-checkpoint gate 18174 loaded the actual
step-250 weights under that exact mode and completed in 00:04:09. It produced 16/16 non-empty,
non-thinking, pairwise-distinct Qwen3-4B outputs: 12 stopped naturally and four reached the
512-token cap, with 413--512 completion tokens. This is the mode frozen for C3 evaluation.

The first old-spooled primary-matrix tasks B1 (18097_0) and B3 (18097_4) failed in two seconds
while concurrent jobs raced updating the shared remote Git ref; neither reached environment setup
or training. Fresh bounded resume array 18195 contains both repository and per-environment `flock`
protection, depends `afterany` on the entire primary array, and will rerun incomplete artifacts
while preflighting completed ones. It supersedes pending array 18117 before any task ran.
After D1 and D2 were safely running, the remaining unstarted old-spooled elements D3/E2/E3/E4
(18097_15--18) were cancelled before execution. Array 18195 contains all four indices, so this
removes the known unlocked-fetch race without omitting any baseline; the primary array's expected
aggregate `CANCELLED` state reflects these superseded pending elements rather than a new run fault.

Job 18097 and the first A0/A1 resume passes wait until C3 job 18098 terminates. This reserves the
all-GPU sequence 18092 -> 18098 before one-GPU work can occupy a released device. Their `afterany`
edges release the other baselines and controls even if C3 fails, while controller 18210 separately
requires C3 success before evaluation fan-out.

Controller 18210 replaces pending controllers 18120, 18125, 18130, 18133, 18189, and 18197. Controller
18133 was cancelled before execution after its spooled script was found to predate the verified C3
BF16 serving contract; 18189 was replaced before execution so every fan-out child is spooled with
the environment-lock repair; 18197 was likewise replaced before execution when its control jobs
were respun with that lock. Gate 18209 reruns exact target validation only after every other
controller prerequisite succeeds; this keeps its Slurm ID fresh when controller 18210 submits
downstream dependencies and avoids relying on purged historical gate 18132. Controller 18210
also requires B4 serving proof 18191 before it submits A1 and historical A3 controls plus 19 trained
model chains. Each trained
model chain has three resumable generation passes, two resumable judge passes, two cached anchored
evaluation passes, and strict checkpoint preflight. The final CPU analysis requires all aligned
evaluation artifacts, runs the frozen prompt-paired contrasts with per-metric Holm correction, and
reports per-prompt favorable/tied/unfavorable directions over every declared clustering threshold.

The final generation preflight also binds every local LoRA or full checkpoint to a streaming
SHA-256 over inference-relevant configuration, tokenizer, index, and weight files. That identity is
stored in the run manifest and must match on every resume, preventing shards from an older model at
the same path from being silently reused. Remote A0/A1 model controls retain their already-frozen
repository revisions and legacy manifests; no local artifact exists to hash for those runs.

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
  production allocation was therefore 12 hours instead of the default six-hour research-job
  bound. The four-GPU smoke and production run both completed, and gates 18167/18174 proved the
  deterministic BF16 deployment path for the full checkpoint.
- The pinned-tokenizer corpus audit found no SFT/OPSD overflow at 3,072 and no prompt or on-policy
  overflow at GKD 2,048. Three of 1,000 historical human targets still require prompt-preserving
  completion-tail truncation at the hardware-safe GKD limit; this 0.3% rate and token count remain
  mandatory diagnostics. DistiLLM's 896-token full-chat proxy exceeds its cap for 1/1,000 records
  by 19 tokens, while all prompts fit its separate 480-token prompt cap.
- GKD on-policy sampling is explicitly overwritten on the official trainer after construction and
  recorded in run metadata. Model defaults are not accepted as implicit experimental controls.
