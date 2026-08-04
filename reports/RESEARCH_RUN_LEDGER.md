# Research run ledger

This ledger freezes the submitted Turing job graph and completed systems evidence as of
2026-08-02 (Asia/Kolkata). Slurm state is live; job IDs and dependency edges are the durable
record. All newly staged jobs use `codex/implementation` and synchronize through
`.git/novelty-distill-sync.lock` before consuming repository code. Primary array 18097 was spooled
before that lock was added; it runs one task at a time while controls occupy the other GPUs, and
locked resume arrays 18195 -> 18212 cover incomplete and newly deferred artifacts.

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
| A1 Qwen3-14B | 18078 | 18162 -> 18198 -> 18199 | 18201 -> 18202 | submitted by controller 18222 |
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
and 18184--18185. Controller 18222 now waits for final score 18202 and control evaluation 18208.
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
| TRL/OPSD bounded resumes | 18195 -> 18212 | indices 0-4, 6-11, and 13-18; chained `afterany`; 25-step checkpoints |
| C3 DistiLLM | 18098 | index 12, four GPUs, 12-hour bound; after target gate 18132 and smoke 18092 |
| Final target replay | 18209 | waits for 18212, final A1 score 18202, and final A0 evaluation 18208 |
| Evaluation fan-out controller | 18222 | waits for final target replay 18209 and B4 deploy gate 18191; names the fresh target gate for every submitted evaluation chain |

Completed TOMATO-1k production runs currently have the following measured systems costs. Training
losses are intentionally omitted from this cross-backend table because CE, GEM, forward/reverse
KL, and skew KL have different numerical scales.

| Baseline | Objective/view | Available rows | Exposures | Train runtime (s) | Peak GPU MiB |
|---|---|---:|---:|---:|---:|
| B1 | CE / human | 1,000 | 1,000 | 305.3 | 15,286 |
| B2a | CE / random-1 | 1,000 | 1,000 | 245.9 | 10,902 |
| B2b | CE / best-1 | 1,000 | 1,000 | 245.1 | 10,982 |
| B2c | CE / mode-1 | 1,000 | 1,000 | 251.0 | 10,924 |
| B3 | CE / diverse-4 | 4,000 | 1,000 | 240.8 | 10,924 |
| B4 | GEM / diverse-4 | 4,000 | 1,000 | 340.9 | 39,626 |
| C1-human | forward KL / human | 1,000 | 1,000 | 617.7 | 45,662 |
| C1-best1 | forward KL / best-1 | 1,000 | 1,000 | 487.2 | 40,298 |
| C1-diverse4 | forward KL / diverse-4 | 4,000 | 1,000 | 485.8 | 40,498 |
| C2-human | reverse KL / human | 1,000 | 1,000 | 617.0 | 43,020 |
| C2-best1 | reverse KL / best-1 | 1,000 | 1,000 | 489.9 | 40,024 |
| C2-diverse4 | reverse KL / diverse-4 | 4,000 | 1,000 | 487.7 | 40,078 |
| C3 | skew KL / best-1 | 1,000 | 1,000 | 1,846.0 | 48,502 |
| D1 | forward KL / student trajectories | 1,000 | 1,000 | 13,524.8 | ~39,500 |
| D2 | reverse KL / student trajectories | 1,000 | 1,000 | 13,638.4 | ~39,500 |

All thirteen completed LoRA artifacts above B4/C3 open as 132,187,888-byte rank-16 adapters with
504 non-empty tensors. B4 is a structurally valid 8,044,981,992-byte, 398-tensor native-BF16 full
model; C3's full-checkpoint and BF16-cast serving evidence is recorded below. The three long human
targets in C1/C2 account for 1,446 completion-tail tokens and materially increase runtime and peak
memory relative to the teacher-target views; this context-cost difference must remain visible in
method comparisons. D1/D2 peak memory is the approximately 38.6-GiB live observation rather than
a profiler-derived maximum, hence the tilde in the table.

The checkpoint-125 trainer histories provide a separate convergence diagnostic. Values below are
25-step window means, so they are comparable only within an objective and are not downstream
quality metrics.

| Baseline | First-window loss | Final-window loss | Change | Final-window gradient norm |
|---|---:|---:|---:|---:|
| B1 | 1.7796 | 1.3845 | -22.2% | 0.3506 |
| B2a | 1.1235 | 0.6400 | -43.0% | 0.4247 |
| B2b | 1.1172 | 0.6485 | -42.0% | 0.4165 |
| B2c | 1.1046 | 0.6466 | -41.5% | 0.4169 |
| B3 | 1.1429 | 0.6514 | -43.0% | 0.4166 |
| C1-human | 3.4759 | 2.5733 | -26.0% | 1.9729 |
| C1-best1 | 3.8937 | 2.5084 | -35.6% | 2.4118 |
| C1-diverse4 | 3.9038 | 2.4845 | -36.4% | 2.3711 |
| C2-human | 3.4503 | 2.9953 | -13.2% | 3.7836 |
| C2-best1 | 3.4373 | 2.8166 | -18.1% | 3.6726 |
| C2-diverse4 | 3.4013 | 2.7882 | -18.0% | 3.7093 |
| D1 | 3.5888 | 2.4294 | -32.3% | 2.0073 |
| D2 | 3.0138 | 2.6429 | -12.3% | 2.9238 |

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
arrays 18195/18212 use the same 116 GiB reservation, allowing their two-task throttle to coexist
with both 64 GiB controls under the node's measured RAM ceiling; this changes only Slurm
reservation accounting, not any batch, optimizer, context, or checkpoint setting. The second pass
is required because D3 first enters the locked graph in 18195 and, at the measured 4.4-minute GKD
step time, needs its own checkpoint-75 continuation. D1/D2 and short offline tasks will preflight
complete in 18212; D3 consumes the remaining optimizer steps.

At 2026-08-02 20:22 IST, D1 and D2 had both produced validated checkpoint-25 directories and were
at steps 27/125 and 25/125 respectively. The same snapshot contained 822/1,658 A1 prompt shards and
1,511/1,658 A0 prompt shards. These are live progress counts rather than completed-result claims.

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
or training. Fresh bounded resume arrays 18195/18212 contain both repository and per-environment
`flock` protection. The first depends `afterany` on the entire primary array, and the second depends
`afterany` on the first; both rerun incomplete artifacts while preflighting completed ones. They
supersede pending array 18117 before any task ran.
After D1 and D2 were safely running, the remaining unstarted old-spooled elements D3/E2/E3/E4
(18097_15--18) were cancelled before execution. Arrays 18195/18212 contain all four indices, so this
removes the known unlocked-fetch race without omitting any baseline; the primary array's expected
aggregate `CANCELLED` state reflects these superseded pending elements rather than a new run fault.

Job 18097 and the first A0/A1 resume passes wait until C3 job 18098 terminates. This reserves the
all-GPU sequence 18092 -> 18098 before one-GPU work can occupy a released device. Their `afterany`
edges release the other baselines and controls even if C3 fails, while controller 18222 separately
requires C3 success before evaluation fan-out.

Controller 18222 replaces pending controllers 18120, 18125, 18130, 18133, 18189, 18197, and 18210.
Job 18210 was cancelled with zero runtime only after replacement 18222 was accepted with the same
upstream dependencies and exports. Controller
18133 was cancelled before execution after its spooled script was found to predate the verified C3
BF16 serving contract; 18189 was replaced before execution so every fan-out child is spooled with
the environment-lock repair; 18197 was likewise replaced before execution when its control jobs
were respun with that lock. Gate 18209 reruns exact target validation only after every other
controller prerequisite succeeds; this keeps its Slurm ID fresh when controller 18222 submits
downstream dependencies and avoids relying on purged historical gate 18132. Controller 18222
also requires B4 serving proof 18191 before it submits A1 and historical A3 controls plus 19 trained
model chains. Each trained
model chain has three resumable generation passes, two resumable judge passes, two cached anchored
evaluation passes, and strict checkpoint preflight. The final CPU analysis requires all aligned
evaluation artifacts, runs the frozen prompt-paired contrasts with per-metric Holm correction, and
reports per-prompt favorable/tied/unfavorable directions over every declared clustering threshold.
Commit `786bfc0` additionally makes controller 18222 submit two resumable research-taste annotation
passes for A0, A1, A3, and every executable trained method. These attributed Chen--Zhao--Cohan
taxonomy results are stored and analyzed separately as `secondary_descriptive`; they cannot modify
the primary contrast family and require the declared two-human agreement gate before any headline
claim. The complete contract is in `reports/RESEARCH_TASTE_PROTOCOL.md`.
Commit `c5cb924` adds the deterministic source-blinded 150-record calibration packet, hidden source
key, and fail-closed two-human agreement analysis. The final CPU job prepares those artifacts but
does not promote automatic labels without Cohen's kappa >= 0.80 on both axes for every automatic--
human and human--human comparison.
Early resumable A0 annotation passes 18227 -> 18228 failed at repository setup during the transient
DNS incident and wrote no labels. Retry-capable passes 18403 -> 18404 replace them and started on
the fourth GPU after both DRKL integration smokes completed. The final controller preflights their
stable output path rather than duplicating completed labels. The taste passes write the same content-bound
namespace that controller 18222 later preflights, so early work is reused and cannot duplicate or
silently conflict with the final graph.

At 21:08 IST on 2026-08-02, A0 generation 18165 completed all 1,658 temporal prompts and released
judge pass 18203. A1 generation 18162 and the D1/D2 online-distillation jobs remained healthy.
Commits `93cd569`--`bc7db5c` add a four-GPU scale path with disjoint generation, judge, and
embedding partitions plus global fail-closed gates and a strict cluster merge. The original 5k
chain 18235--18242 -> 18246 -> 18247 and 20k chain 18248--18257 were cancelled with zero runtime
after the transient-DNS audit showed that their spooled scripts predated the bounded repository
retry. Fresh 5k chain 18380--18389 is gated behind controller 18222; fresh 20k chain 18390--18399
cannot begin until the new 5k target gate 18389 passes. Bootstrap jobs 18380 and 18390 verify prompt
text hashes and frozen generation controls before copying the exact nested 1k -> 5k -> 20k shards;
identical prompts are not regenerated. The cancelled zero-runtime 5k merge/target jobs 18243 and
18244 were replaced after Turing automatically attached a GPU to the original four-CPU merge;
the new merge jobs preserve the corrected two-CPU request.

The final generation preflight also binds every local LoRA or full checkpoint to a streaming
SHA-256 over inference-relevant configuration, tokenizer, index, and weight files. That identity is
stored in the run manifest and must match on every resume, preventing shards from an older model at
the same path from being silently reused. Remote A0/A1 model controls retain their already-frozen
repository revisions and legacy manifests; no local artifact exists to hash for those runs.
The node01 audit produced identities `7fc74fcd...b50f8` for the C1-best1 adapter,
`6dc6fb9c...1e949c` for B4, and `c6dee08e...1f72b` for C3. Streaming the two approximately 8 GB
full checkpoints took 5.2 and 5.7 seconds, so this gate is negligible beside model startup.

The registry contains 24 planned entries: four non-training controls, nineteen executable training
variants, and E1. E1 is explicitly `fail_closed` because the pinned official OPSD implementation
does not expose static-trajectory off-policy training. It is neither missing nor silently skipped,
and it cannot enter the executable matrix without a reviewed upstream implementation.

Commit `1bfddd0` adds deterministic offline W&B backfill from immutable training, evaluation, and
analysis JSON rather than making credentials or a remote service part of training correctness.
Commit `916a1bd` routes W&B run, data, cache, and configuration directories to scratch. A real
W&B 0.22.3 B1 export on node01 wrote the config, numeric summary, metric table, and metadata
artifact; an immediate repeat exported zero records and left exactly one offline run directory.
Final primary and DRKL analysis jobs now require their export manifests before succeeding.

### 2026-08-03 transient DNS recovery

The overnight A1 generation job 18162 reached 1,110/1,658 prompts before its six-hour limit, while
D1 and D2 reached valid checkpoint 75 before their limits. Their first continuation jobs failed
during repository setup because the login service temporarily could not resolve `github.com`;
they did not alter model or generation artifacts. Commit `33bbbae` replaces one-shot fetches in
future submissions with a bounded six-attempt, lock-protected repository synchronizer that refuses
to continue from a stale commit.

Recovery jobs 18357 -> 18358 resume only A1's 548 missing prompt files. Array 18359, followed by
one bounded second pass 18360, preflights every existing baseline and resumes only incomplete work.
B1 and B3 completed on the first recovery pass, and D1/D2 resumed from their preserved checkpoints;
D3/E2/E3/E4 remain in the same array. The downstream target-validation and evaluation chain was
rewired to these replacements without weakening any `afterok` gate. No scientific output was
accepted merely because a Slurm job exited successfully: content-bound manifests and
`run_metadata.json` completeness checks remain authoritative.

Commit `98e4a53` also prepares, but does not mix into this recovery, a separate Qwen3-1.7B student /
Qwen3-8B teacher replication profile. Its teacher samples, targets, checkpoints, job identities,
and analysis namespace are disjoint from the main Qwen3-4B / Qwen3-14B study and will run only
after the main-stage analysis gate.

After the primary matrix had been frozen, Luong, Tran, and Chen's DRKL objective was identified as
a directly relevant secondary treatment for the reverse-KL diversity mechanism. Commit `b20bbf0`
implements Equation 9 independently, verifies the value and gradient against the direct definition
in the cluster Torch environment, and keeps `F1-best1`/`F1-diverse4` in a separate registry. Low-
priority jobs 18377 and 18378 are staged `afterok:18222`; they cannot delay or enter the frozen
primary matrix. Their matched controls and interpretation boundary are recorded in
`reports/EXPLORATORY_TREATMENTS.md`. Low-priority controller 18379 waits for both F1 checkpoints,
then submits their resumable temporal generation, scoring, evaluation, and research-taste chains.
Its final analysis additionally waits for the primary analysis job recovered from controller
18222, guaranteeing that the matched C2 and A1/A3 artifacts are complete before any F1 contrast.
One-step integration jobs 18400 and 18401 then exercised the actual TRL subclass on the fourth GPU
while the three primary recovery jobs ran. Best-1 completed in 20 seconds with finite loss
1.0041895; diverse-4 completed in 17 seconds with four available rows, one budget-matched exposure,
and finite loss 0.9756892. Both recorded `diversity_aware_reverse_kl` and gamma 0.5 and produced
readable LoRA adapters. This verifies both end-to-end implementation paths; it is systems evidence
rather than an ideation result.

Research-taste jobs 18403 and 18404 exposed a deterministic structured-output edge case: the
32B annotator returned one whitespace-heavy JSON object truncated at the original 192-token cap,
so the concurrent batch stopped after preserving two complete prompt shards. The annotation
contract now allows 512 output tokens and performs three bounded request-level retries while
retaining the same pinned model, zero-temperature decoding, and resumable shard validation. The
generic JSON-schema grammar still allowed unbounded inter-field whitespace, so the final serving
contract uses an equivalent compact regex grammar that fixes field order, enum values, booleans,
and score ranges without permitting whitespace loops. This is a serving-reliability change, not a
taxonomy or outcome change.
After the incompatible diagnostic shards were moved to the recoverable
`research-taste-superseded` namespace, job 18415 crossed the formerly failing third prompt under
the compact grammar without a retry and continued writing validated shards; job 18416 is its
bounded resumable second pass.

The final primary-analysis job now begins with a fail-closed training-matrix audit over all 19
runnable methods and records E1 separately with its frozen non-executable reason. The audit
recomputes the live training-input and registry hashes, requires the identical dataset revision,
ordered example-ID set, student revision, seed, and optimizer-example exposure count across
methods, verifies expected one-versus-four-target row counts, checks backend completion evidence,
and hashes every deployable model artifact. The final W&B snapshot includes this audit artifact;
no result matrix can be published from missing, stale, underexposed, or byte-incompatible training
runs. OPSD metadata now records its divergence, trajectory source, and target view before the
pending E2/E3/E4 production tasks start.

Commits `31f7022` and `84d07dc` bind each promoted method/seed to its exact array task, model or
adapter bytes, temporal evaluation, secondary research-taste annotation, scale-relative controls,
and balanced final analysis. Replication manifests now serve Qwen3-1.7B under its true pinned
identity rather than inheriting the 4B evaluation config. The optional final-stage branch runs all
100 curated NoveltyBench prompts plus the complete 61/9/35-case causal/3D/Boolean HypoSpace suite
at K=10, requires all four combined artifacts, and reports descriptive seed-balanced means and
variation. It is intentionally omitted from the 5k selection stage. Commit `9413999` adds a
separate one-time replication-control graph: untouched Qwen3-1.7B and Qwen3-8B teacher outputs are
generated and scored independently at K=16 and receive their own research-taste annotations; no
4B/14B behavioral control is reused.

At 20:40 IST on 2026-08-03, recovery tasks 18359_13 and 18359_14 wrote and passed direct integrity
checks on D1 and D2 checkpoint 100. Both trainer states record exactly 100/125 steps, epoch 0.8,
finite loss/gradient histories, and readable 132,187,888-byte adapters. D1's step-100 loss and
gradient norm are 2.4526 and 2.0063; D2's are 2.7276 and 3.1619. These are optimization telemetry,
not evaluation outcomes. The same snapshot contained 1,307/1,658 A1 temporal generation shards
and 310/1,658 validated A0 research-taste shards. Jobs 18358 and 18416 remain bounded resume passes
for the unfinished stable namespaces.

The interpretation protocol requires a matched exposure check before promoting a four-reference
finding. Commit `54f242a` therefore freezes a separate five-run sensitivity: B2a/B2b/B2c repeat
their single target to exactly 4,000 optimizer-example exposures, while B3/B4 consume the full
four-target pool at the same 4,000-exposure budget. It is excluded from the primary registry and
Holm family. Low-priority arrays 18452 -> 18453 are staged behind primary controller 18222 with a
two-GPU throttle and bounded resume behavior; they cannot consume a GPU before the frozen primary
fan-out is released. The claim boundary and comparisons are in
`reports/EXPOSURE_SENSITIVITY_PROTOCOL.md`.

The five sensitivity checkpoints now have complete, separately named temporal evaluation and
research-taste chains. Their terminal evaluation jobs are 18462, 18471, 18480, 18489, and 18498;
their terminal taste jobs are 18464, 18473, 18482, 18491, and 18500. Because primary controller
18222 can reveal its final analysis job only after it executes, commit `c7d6bee` adds a fail-closed
deferred handoff. Job 18502 waits for controller 18222, parses the controller's last valid JSON
analysis record, and submits the exposure analyzer behind that primary analysis plus all ten
sensitivity endpoints. It atomically records the resolved dependency graph in
`submissions/exposure-sensitivity-analysis.json`; malformed, missing, or incomplete IDs terminate
the controller without producing a finding.

At 21:05 IST on 2026-08-03, D1 and D2 had both reached step 105/125 with finite telemetry. A1's
current temporal-generation pass had completed 230 of the 548 shards missing at pass start
(1,340/1,658 total), and A0 research-taste annotation had completed 423/1,658. These four jobs
occupied all four GPUs. The TasteGap source audit was repeated against arXiv v1, the author code,
and the IdeaSeed card: our two seven-label axes and TVD/base-2-JSD/normalized-entropy definitions
match the paper's primary distributional analysis, while our human gate is deliberately stricter
because it also requires human--human kappa >= 0.80. Neither the author repository nor IdeaSeed
declares a reuse license, so only the independently implemented taxonomy analysis is active; the
released dataset is not silently added to the official benchmark suite.

Before any trained-model temporal metric existed, commits `a8512cc` and `c957b23` froze a second
paper-derived diagnostic from TasteGap's same-paper geometry analysis. It reuses only the exact
normalized Qwen3 embedding cache already produced by evaluation and measures, per prompt, each
method's cosine affinity to A1, affinity to A3, teacher-minus-human affinity, and within-method
concentration. Ten-thousand-resample prompt bootstraps and A0 deltas are descriptive; missing cache
vectors fail rather than triggering new inference. The path is wired into primary, exposure, DRKL,
and promoted analysis jobs and specified in `reports/SAME_PROMPT_GEOMETRY_PROTOCOL.md`.
The analyzer keeps only A1/A3 plus one candidate's float32 arrays resident and batches bootstrap
indices; a full 1,658-prompt, 23-method, 10,000-resample synthetic summary completed locally in
10.86 seconds with 120,924 KiB maximum RSS. Real runtime will therefore be dominated by validated
cache-file reads rather than bootstrap memory.

At 21:40 IST on 2026-08-03, the paper review was extended to Amin et al., *Escaping the Mode
Lottery: Multi-Response Training Improves Language Model Generalization*, arXiv:2606.00544v1.
Its separation of response multiplicity from response selection exposed a confound in the original
five-run exposure study. Commits `7cc4edc` and `ee6ac33` therefore add `F2-random4`, an independent
value-blind Random-4-of-8 implementation derived from the immutable teacher bank. The matched
`F2-random4` versus `B2a-4x` contrast isolates multiplicity; `B3-4x` versus `F2-random4` isolates
coverage-aware selection. This amendment was frozen before any trained-model temporal metric was
available and remains secondary. The cited author's public repository had no explicit license at
commit `8f389da41ad3c8441c9569f746a9a040cf388ddc`, so no external code was copied.
Commit `43c7d39` subsequently made the uniform random-4 subset nested around B2a's exact random-1
target, reducing selector noise without changing the Random-4 distribution. F2 was still pending
behind controller 18222 and had produced no training artifact when this refinement was frozen.
Commit `bad86f6` completes the secondary selector graph with direct `B2b-4x` and `B2c-4x`
comparisons against `B2a-4x`. The matched edges now isolate best/reward selection and modal
selection at K=1, multiplicity under unbiased selection, and coverage-aware selection at K=4.

The low-priority F2 chain is now staged behind primary controller 18222: training jobs 18519 ->
18520, generation terminal 18523, scoring terminal 18525, evaluation terminal 18527, and
research-taste terminal 18529. Replacement controller 18530 joins all six exposure evaluation and
taste terminals to the future primary analysis. The superseded five-method controller 18502 was
canceled while pending with exactly zero runtime, after 18530 was accepted and its primary
dependency audited. At the same snapshot, D1 and D2 were both at step 113/125, A1 temporal
generation contained 1,407/1,658 validated shards, and A0 research-taste annotation contained
582/1,658 shards. These four active jobs still occupied the available GPUs; none of the newly
staged low-priority work had begun.

At 21:49 IST, a backend-aware partial training audit found a historical metadata-schema gap before
it could reach final analysis: the six completed C1/C2 TRL-GKD runs predated commit `b20bbf0`, which
began writing the declared `divergence` field. Their immutable training records already contained
the exact identifying controls (`lambda=0, beta=0` for forward KL and `lambda=0, beta=1` for
reverse KL), but the new fail-closed audit correctly refused to infer a missing declaration.
Commit `6a8b2f3` adds a narrow idempotent migration. It preserves each original metadata file
byte-for-byte as `run_metadata.pre-divergence-backfill.json`, records its prior SHA-256, the live
registry SHA-256, the proving lambda/beta values, and the migration commit, then atomically adds the
declared field. Ambiguous or conflicting evidence fails rather than mutating metadata.

The migration manifest is
`audits/tomato1k-divergence-backfill.json`. After migration, the formal subset audit at
`audits/tomato1k-training-partial-13.json` passed all 13 completed methods. It proved a common
contract of dataset revision `fcd201d92758a642465a7653b8055f3a04d5f439`, 1,000 ordered examples
(ID hash `6ce87308...`), identical 6,767,710-byte input (SHA-256 `995c9648...`), pinned Qwen3-4B
revision `1cfa9a7...`, seed 17, and 1,000 optimizer-example exposures, and it content-hashed every
deployable artifact including the full GEM and DistiLLM checkpoints. At this snapshot D1 and D2
were both at step 115/125, A1 temporal generation had 1,422/1,658 shards, and A0 research-taste
annotation had 623/1,658 shards.
Commit `fa1ae14` also places the migration ahead of the final full-matrix audit. It treats native
current-schema metadata as `not_required`, legacy migrated metadata as `already_migrated`, and
missing legacy declarations as a one-time backfill; any mismatch still terminates analysis.

At 22:00 IST, the preregistered teacher-target diagnostic for the nested random-4 control completed
without GPU inference. Its immutable result is
`evaluations/teacher-targets-random4-teacher-1k-v1/analysis.json` on scratch, produced by commit
`033ca707`. Random-4 preserved random-1 mean quality (0.826950 versus 0.827300) while increasing
automatic semantic modes per prompt from 1.000 to 2.584. Diverse-4 reached 0.866088 quality and
3.146 modes, so it changes both response multiplicity and target selection. This pre-student
finding validates the six-arm exposure design: F2 versus repeated B2a isolates unbiased
multiplicity, while B3 versus F2 isolates coverage/quality-aware selection. Full provenance,
deltas, and interpretation limits are recorded in `reports/RANDOM4_TARGET_FINDINGS.md`.

Between 22:00 and 22:34 IST, three additional 2026 preprints were screened before student metrics
were available. Progressive conditional surprise was retained only as a possible post-promotion,
embedding-independent collapse diagnostic; IdeaGene-Bench sharpened the human gate to distinguish
topical distance from coherent mechanism inheritance and limitation repair. IDEAgent's joint
quality-diversity Yield motivated commit `10b4d51`, which independently implements a secondary
`viable_semantic_yield` over the existing K=16 samples. It gates relevance, soundness, and clarity
at 4/5, then exactly finds the largest subset whose every pair is below the current cosine
threshold. It is reported across the full threshold curve and remains outside the primary Holm
family because automatic non-obviousness is deliberately absent. Commit `f7f1e31` preserves the
gate definition in offline tracking provenance. The full repository suite passed with 262 tests
and two expected local PyTorch-dependent skips. None of the three author repositories supplied a
usable code/data license at the audited commits, so no external implementation or dataset was
imported.

At 22:34 IST, D1 and D2 completed all 125 on-policy TOMATO-1k steps with clean Slurm exits. D1
(forward KL, student trajectories) recorded train loss 0.969608 and runtime 13,524.76 seconds;
D2 (reverse KL, student trajectories) recorded train loss 1.060318 and runtime 13,638.43 seconds.
Both used the common 1,000-example/1,000-exposure, seed-17 contract. Because they launched from
pre-schema commit `98e4a538`, their exact original metadata was preserved and the explicit
divergence field was migrated using the proving `(lambda, beta)` pairs `(1, 0)` and `(1, 1)`.
The migration manifest is `audits/tomato1k-divergence-backfill-d1-d2.json`.

The formal 15-method subset audit then passed at
`audits/tomato1k-training-partial-15.json`. It content-hashed the D1 adapter as `47a46794...` and
the D2 adapter as `56904eba...` and reproved the same dataset revision, ordered-ID hash, input
bytes, Qwen3-4B revision, seed, and exposure budget across all completed methods. This is training
completion evidence, not a performance comparison. The freed slots immediately started D3 job
18559 and E2 job 18561 from repository commit `f7f1e31`; E3 and E4 remain next in the bounded
array. At the same snapshot, A1 temporal generation contained 1,490/1,658 validated prompt shards
and A0 research-taste annotation contained 838/1,658.

The measured first-step runtimes imply that D3 cannot finish inside its inherited six-hour Slurm
limit and E2 is close to that boundary. Slurm would not extend the already-running array tasks, but
it accepted a 12-hour limit for the not-yet-started E3/E4 tasks. Targeted recovery array 18568
(`15-16%2`, dependency `afterany:18559:18561`) was therefore staged to resume D3/E2 as soon as
their current allocations end, rather than waiting for the original whole-array recovery 18360.
The original recovery remains an idempotent safety net. This scheduling change does not alter data,
optimizer, seed, checkpoint, or evaluation contracts; it only closes otherwise idle GPU time.

The complete D1/D2 checkpoint-125 trainer histories contain exactly 125 finite, contiguous loss
and gradient records. D1's first-to-final 25-step mean loss falls from 3.5888 to 2.4294 and its
first-to-final 10-step mean gradient norm falls from 13.0524 to 1.9936. D2's corresponding loss
means fall from 3.0138 to 2.6429 and gradient means from 4.0442 to 2.8803. D1's higher early
variance and lower late loss than D2 are optimizer behavior under different divergence objectives,
not evidence that D1 generates better ideas; only the frozen temporal evaluation can establish
that. The minimum observed step losses occur late in both runs (D1 step 96, D2 step 109), while
both maxima occur in the first four steps, providing a basic convergence sanity check.

At 23:00 IST, the first secondary A0 research-taste pass was stopped after a fail-closed interface
audit. Although all 933 completed shards were structurally valid with unique request IDs, more than
85% of their opportunity labels collapsed onto scope mismatch and a known mechanism-specific
example received bottleneck specificity 1/3. Re-querying the same pinned Qwen3-32B-FP8 server on a
label-stratified 20-record diagnostic sample without the compact label regex produced only 45%
opportunity-label agreement and raised mean specificity from 1.05 to 2.75. This isolates a strong
annotation-interface effect; it does not establish that the alternative labels are correct.

Only jobs 18415 and its pending resume 18416 were cancelled. Their outputs were preserved under
`research-taste-invalid-compact-v1/A0-temporal-k16-seed17000` and are excluded from analysis. The
replacement protocol removes forced label decoding, adds the source paper's decision guidance,
strictly accepts raw or singly fenced JSON, and receives a new protocol hash. Primary training,
A1 temporal generation, and all frozen quality/semantic-mode outcomes were unaffected.

Replacement live smoke 18583 subsequently failed closed in 00:01:26 before writing any shard. The
unconstrained model used `boilerplate`/`diagnostic_score` aliases instead of the exact six-field
schema; strict validation rejected all attempts. Its automatically released resume 18584 was
cancelled after 12 seconds. A new red--green regression requires the literal six-key contract in
the annotator prompt before the next submission. No partial replacement output exists.

Corrected replacement job 18585 started from commit `d8c89f3` with a 12-hour limit, followed by
idempotent resume 18586. Its first permanent shard passed the full current-protocol validator at
hash `55d9ba2b...`, with all 16 records, 16 unique request IDs, exact diagnostic keys, and no retry.
This proves the repaired deployment path, not taxonomy accuracy; human calibration remains the
claim gate.

After 35 repaired A0 prompt shards, an ordered partial integrity screen found opportunity labels
unanimous within 82.9% of prompts and method labels unanimous within 54.3%. These are not outcome
estimates. They exposed a construct risk specific to transferring TasteGap onto TOMATO: an explicit
mechanism question may determine the opportunity label before the response does. The secondary
summary and Markdown report now freeze mean normalized within-prompt entropy and prompt-unanimity
rate on both axes. Near-zero conditional entropy must qualify any global taste-distance claim; A3's
K=1 values are explicitly marked mathematically degenerate. This diagnostic was added before any
trained-model temporal generation existed.

At 23:22 IST, the four live GPU allocations remained healthy: A1 temporal generation had 1,577 of
1,658 prompt shards, repaired A0 research-taste annotation had 67 of 1,658 prompt shards, D3 had
reached step 11/125, and E2 had reached step 16/125. Slurm RSS accounting for the two live trainers
was approximately 1.6 and 1.8 GiB, respectively. Pending E3/E4 tasks 18359_17--18 and targeted
D3/E2 recovery tasks 18568_15--16 originally reserved 116--128 GiB each, which would have prevented
all four GPU tasks from coexisting despite the measured host-memory headroom. Their pending-only
host-memory requests were reduced to 70 GiB each. Slurm reread the four requests as 70 GiB while
preserving their GPU, time-limit, dependency, configuration, checkpoint, seed, data, and optimizer
contracts. This is a scheduling correction, not an experimental change.

CPU-only replay job 18596 then revalidated the complete canonical TOMATO family directly from the
saved artifacts. Across both open and composition tasks it checked 1,000-, 5,000-, and 20,000-row
training sets plus the 1,658-row test set; every row schema, dataset revision, prompt hash, paired
task ID set, strict train-set nesting relation, and train/test disjointness check passed. The replay
manifest at `audits/tomato-family-manifest-replay-20260803.json` is byte-identical to the original
`data/tomato-family-manifest.json`, with SHA-256 `4a8f3f815d966aee2eb711b702b6e9cf958a75c238505ae38139e5692b456301`.
Initial job 18595 was cancelled while pending at zero elapsed time because Turing automatically
added GPU billing to its four-CPU request; the two-CPU replacement completed in three seconds with
an empty `SLURM_JOB_GPUS` field and did not displace research compute.

That observed scheduler behavior also exposed five not-yet-released CPU-only scripts requesting
four cores: primary, promoted, exposure-sensitivity, and DRKL analyses plus the historical-control
preparation job. They now request two cores, below Turing's automatic GPU-billing threshold. A
repository-wide Slurm contract test rejects any CPU-only batch script above that threshold; the
full suite passed with 268 tests and two expected local skips. No already-running allocation or
experimental contract changed.

The first 18 E2 optimizer steps then exposed an upstream optional-loss defect: every finite logged
loss was negative under `jsd_token_clip: 0.05`. Source tracing showed that the pinned OPSD trainer
upper-clips signed per-vocabulary KL/JSD contributions before summing them. Only their vocabulary
sum is guaranteed nonnegative, so capping positive contributions while leaving negative ones
uncapped invalidates the divergence. Job 18561 was cancelled before checkpoint 25; its three
trajectory logs were preserved at
`checkpoints-invalid-vocab-clip-v1/E2-tomato1k-seed17-job18561`, and no invalid optimizer state is
reused. Pending E3/E4 and E2 recovery tasks were held before the cancellation.

Commit `c0f9615` sets the optional clip to `null` in all OPSD configurations while retaining the
pinned official unclipped loss. The configuration and fail-closed run-spec tests failed before
their respective changes, and the full repaired suite passed with 267 tests plus two expected
local skips. Qwen3-4B gate 18597 then
completed a real corrected E2 update with loss 1.215773, gradient norm 2.714494, 23.542-second
runtime, saved `jsd_token_clip: null` metadata, and a loadable adapter. Corrected E3 task 18592 was
released next; fresh E2 task 18568_16 and E4 remained scheduled behind available resources and the
array throttle. The algebraic counterexample, evidence, and interpretation limits are frozen in
`reports/OPSD_LOSS_VALIDITY_AUDIT.md`.

CPU-only score-diagnostic job 18600 then validated all 1,658 A0 temporal score shards from judge
jobs 18203--18204 and summarized 26,528 generations. The composite quality proxy is 0.933896
(population SD 0.065451), with a 0.253920 ceiling rate and mean within-prompt range 0.130157. Only
six generations reached the length limit (0.000226), and the prompt-centered quality--length
correlation is 0.0701. Rubric discrimination is uneven: rating-5 rates are 0.9992 for compliance,
0.9588 for clarity, 0.9263 for relevance, 0.6412 for soundness, and 0.2541 for feasibility. Final
comparisons must therefore retain dimension-level effects rather than relying on the near-ceiling
composite alone. This is control-only descriptive evidence; semantic modes and paired model
contrasts remain gated. Full provenance and interpretation are in
`reports/A0_TEMPORAL_SCORE_FINDINGS.md`.

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

## 2026-08-03 research-taste axis-swap recovery

Research-taste annotation job 18585 failed after completing 135 valid A0 prompt shards because
four samples for prompt `2025_41085166` repeatedly returned an opportunity label in the method
field and a method label in the opportunity field. Resume job 18586 was held before consuming a
GPU. The parser now repairs only the lossless case where both supplied labels are valid members of
the opposite, disjoint taxonomy axes; partial or unknown mismatches still fail closed. Every
repair is persisted as `axis_swap_repaired: true`, summarized as an axis-swap repair rate, and
surfaced in the Markdown report. Existing valid shards remain schema-compatible with a default
false diagnostic. Focused parser, shard, summary, and report tests cover the change; the 135 valid
shards are retained for deterministic resume rather than recomputed.

The live dependency audit then found that the original second-pass training array could eventually
launch D3/E2 tasks against the same deterministic checkpoint directories as targeted recovery
array 18568. The two still-pending second-pass tasks were split by Slurm and changed to wait for
the corresponding targeted recovery (`18360_15` after `18568_15`, `18360_16` after `18568_16`),
after which their existing completion preflight will make them no-ops. This changes scheduling
only. As defense in depth, `train_smoke.sbatch` now takes a nonblocking lock keyed by the resolved
`run_metadata.json` path before its completion check and retains it through process exit. A
misconfigured retry therefore fails explicitly instead of concurrently writing an adapter or full
checkpoint. The behavior was added test-first and the batch script passes Bash syntax validation.

## 2026-08-04 A1 temporal generation completion

Teacher generation job 18357 completed cleanly in 05:30:09, bringing the frozen A1 temporal run to
all 1,658 prompt shards and 26,528 Qwen3-14B samples. Idempotent resume job 18358 then validated the
complete tree and exited successfully in 12 seconds without regenerating a prompt. Qwen3-32B-FP8
score pass 18201 started immediately, with pass 18202 retained as an `afterany` resumable second
pass. This is a complete generation-population gate, not yet an A1 quality or mode finding.

Before any trained-student temporal output was available, a focused literature update added one
interpretation rule without changing the metric family: raw diversity loss is read beside the
already frozen `viable_semantic_yield`. Stable viable yield is compatible with removal of weak
outputs, whereas decreases in both raw coverage and viable yield are evidence of residual
narrowing among operationally viable responses. Karouzos et al. (2026) motivate this distinction;
NovBench independently reinforces the existing rule that the quality judge is not a novelty
oracle. No new training, scorer, endpoint, contrast, or promotion criterion was introduced.

## 2026-08-04 D2 on-policy reverse-KL extension completion

D2's audited checkpoint-125 adapter (`sha256:56904eba...`) was evaluated end to end under the
same frozen K=4 temporal contract as the compact baseline study. Four L40S jobs generated all
6,632 hypotheses for 1,658 held-out TOMATO prompts; the Qwen3-32B-FP8 judge produced all 6,632
rubric records, and Qwen3-Embedding-4B embedded the new outputs against the shared teacher modes
and training-reference targets. Generation and score validators each reported 1,658 complete
prompt shards with zero pending. Evaluation job 18755 and analysis job 18756 exited 0, and the
final queue audit was empty.

The three D2 contrasts were committed before D2 held-out outputs were inspected and form a new
Holm family, leaving the compact study's original five contrasts unchanged. D2's feasibility
(4.2307) and teacher-mode recall (0.1254) were not distinguishable from D1, C2-best1, or C1-best1.
Its soundness was about 0.018--0.020 lower than all three, with Holm-corrected p < 0.04. The large
effect is breadth: viable semantic yield was 1.7370, lower than D1 by 0.4855 (95% CI
[-0.5283, -0.4415]), lower than C2-best1 by 0.0579 ([-0.0971, -0.0181]), and lower than C1-best1
by 0.5109 ([-0.5537, -0.4662]). The first and third corrected p-values are 0.000300; the second
is 0.00470. Thus on-policy reverse KL is a high-quality but strongly mode-seeking control and does
not displace C1-best1 as the practical trained baseline.

The deployment required two transparent operational recoveries. Three initial generation shards
exited immediately during a concurrent repository-clone race and were rerun without accepting
partial output. The first judge pass exhausted transient L40S activation memory when configured
at concurrency eight after 100 valid shards; idempotent recovery at concurrency two preserved the
same judge, revision, prompts, rubric, decoding, and deterministic output contract. Full provenance,
job IDs,
hashes, and interpretation limits are frozen in
`reports/compact-k4-seed17-d2-extension/RUN_AUDIT.md`.
