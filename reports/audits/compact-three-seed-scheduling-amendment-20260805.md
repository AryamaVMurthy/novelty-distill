# Compact three-seed scheduling amendment — 2026-08-05

This note records a Slurm resource-reservation change made after the seed-29
offline runs completed and its two on-policy runs started. It does not change
any scientific run parameter.

## Reason

Seed-29 D1 and D2 each reserved 128 GiB of node01 host memory. Their concurrent
reservations left too little schedulable memory for the seed-43 array even
though two L40S GPUs were idle. Completed seed-29 C1/C2 jobs used approximately
6.0 GiB and 2.3 GiB maximum resident host memory, respectively. Earlier
on-policy runs also used only a small fraction of their 128 GiB reservation.

## Change

- Pending seed-43 training tasks were first reduced from 128 GiB to 70,000 MiB.
- After one seed-43 task started, the still-pending short offline tasks
  (B2b, C1-best1, and C2-best1) were reduced to 32,000 MiB, allowing a fourth
  GPU to become schedulable.
- The still-pending seed-43 D1 and D2 tasks were reduced from 70,000 MiB to
  50,000 MiB. Together with the two 128 GiB seed-29 reservations this remains
  below node01's physical-memory capacity and allows all four long on-policy
  replications to overlap once the short offline tasks finish.
- Pending idempotent final-preflight tasks for seeds 29 and 43 were reduced to
  32,000 MiB because they either validate an already-complete checkpoint or
  resume the same bounded trainer under the checkpoint lock.
- The pending B2a NoveltyBench job was reduced from 96 GiB to 64,000 MiB. The
  completed matched A0 NoveltyBench job used about 25.2 GiB maximum RSS.
- Pending D1 and D2 NoveltyBench jobs on node02 were likewise reduced from
  96 GiB to 64,000 MiB. This allowed D1 to occupy the fourth otherwise-idle
  node02 GPU while B2b, C1-best1, and C2-best1 continued. D2 retains its
  original dependency on B2b and will replace it when that run terminates.
- B2a's seed-17 final LoRA was staged from node01 to an initially absent path
  on node02. The exact inference-artifact identity matched on both nodes:
  `sha256:812cb0cbb2b8bc6efbd5163aeb5c54b2c74b0cd4f400ba8b703cd7a62b2aa7f5`.
  Pending job 19202 was then moved from node01 to node02 so it can replace a
  completed benchmark rather than waiting behind four long training jobs.

Slurm splits individually updated array elements into new concrete job IDs;
their stable identities remain the original array/task pairs (`19111_*`,
`19112_*`, and `19113_*`). Dependencies refer to those stable array/task
identities and remain intact.

## Node02 evaluation staging

To permit a later even split of the twelve seed-29/43 TOMATO evaluation chains,
the immutable shared inputs were copied from node01 to previously absent paths
on node02. Canonical tar-stream hashes (sorted names, normalized timestamp and
ownership) matched across nodes:

| Relative path | Canonical SHA-256 |
|---|---|
| `data/tomato-open-test-1658.jsonl` | `331a5d07bb11c32bb4e54f19782429712d80eef7e45dc88402ac9480bca6f6ee` |
| `data/teacher-targets-tomato1k-v1.json` | `beca2ad13385cf3e4a481f6be89c4b8abffeac9f060934289cafb0806437c585` |
| `generations/evaluation-teacher/A1-temporal-k16-seed17000` | `9bf87b0d1327b1b58bd3b7c03cfe5f0e7f2040afcdfb463485f04a37c5ff8a4a` |
| `evaluation-scores/A1-temporal-k16-seed17000` | `da21a0430386373d2334e8aed996224079ac32e62d0c709b58a5c5ee406c3d1a` |

No generation or score job was moved at this point. Method checkpoints will be
staged and pending chains reassigned only after their training gates pass.

### First four staged chains

At approximately 05:00 IST, four offline-training tasks had complete final
adapters while the four longer on-policy tasks continued. Those four adapters
were copied to previously absent node02 paths and verified with the same
inference-artifact hash on both nodes:

| Chain | Adapter identity | Jobs pinned to node02 |
|---|---|---|
| seed 29 B2a | `sha256:cf129f11e7b6e0227832efd26b3b902b3a6dc492ec19ef4bd408c370078907b8` | 19117--19121 |
| seed 29 C1-best1 | `sha256:9f7974619596b994ecce0f58517ebe0d9f40215c98b56748966b975c96c0a25f` | 19127--19131 |
| seed 43 B2b | `sha256:e76540925b835a62524e5ded1427f3d6196a2088070fef7881e2c8a9bebf0344` | 19152--19156 |
| seed 43 C2-best1 | `sha256:4db42299596af95472c43b0c11f0ee7c671f0c2a16bee1c66b8d5793ca4cd551` | 19162--19166 |

Every listed generation, generation gate, scoring array, scoring gate, and
evaluation job remained pending behind its original dependency. Only
`ReqNodeList` changed from node01 to node02. The remaining planned node02
chains, seed-29 D1 and seed-43 D2, cannot be staged until their on-policy final
adapters exist.

### On-policy checkpoint gate

At approximately 06:16 IST, all four long on-policy tasks had crossed and
durably written checkpoint step 25 of 125. Each checkpoint contained adapter
weights, optimizer and scheduler state, RNG state, adapter configuration, and
trainer state recording `global_step: 25`, `max_steps: 125`, and `epoch: 0.2`.
The jobs then continued from step 26. These are recovery checkpoints only:
downstream evaluation remains gated on the final adapter and will not consume
an intermediate checkpoint.

At approximately 08:09 IST all four tasks also crossed checkpoint step 50.
Each adapter weight file was present at the expected checkpoint directory and
had the common expected size 132,187,888 bytes; their distinct SHA-256 values
were:

| Run | Checkpoint-50 adapter SHA-256 |
|---|---|
| seed 29 D1 | `a3a3c60be202af3847a80c5a348156b2ce80829789d65b2200bf99efd1f528b7` |
| seed 29 D2 | `0dd7258fdf98842a704f1069a2f22fa5e0d4b787d784a00074961f85b2bfb954` |
| seed 43 D1 | `5fb476eceb754086010c708d8616f8245524b58211448723c56ba604a5baf8a2` |
| seed 43 D2 | `94ac5046a3aae87cff1fea50a587fea1b5d1309e74c113419a7e6a46e1619c47` |

Optimizer, scheduler, RNG, trainer-state, tokenizer, and adapter-configuration
files were also present as in checkpoint 25. The tasks continued toward the
frozen final step 125.

### Final-adapter and result-repatriation holds

Two release races were closed while every affected job was still pending:

- generation roots 19137 (seed-29 D1) and 19172 (seed-43 D2) were user-held;
  each complete five-job chain was pinned to node02. A root will be released
  only after the corresponding final adapter is copied from node01 into its
  exact declared path on node02 and its inference-artifact hash matches;
- aggregate analysis job 19177 was user-held on node01. It will be released
  only after all node02 evaluation JSON files are copied into the exact paths
  expected on node01 and their byte hashes match.

These holds impose storage-availability gates only. Dependencies, commands,
seeds, checkpoints, samples, judge, and statistical analysis remain frozen.

### Early release of completed offline-method evaluations

The second-pass training array had an array-wide `afterany` dependency, so its
already-complete offline tasks could not no-op until the four long on-policy
tasks also terminated. This left node02 idle despite final, staged adapters.
Before changing any evaluation dependency, node02 verification allocation
19249 rechecked all four declared adapter paths, all required immutable inputs,
and these inference-artifact identities:

| Chain | Reverified adapter identity |
|---|---|
| seed 29 B2a | `sha256:cf129f11e7b6e0227832efd26b3b902b3a6dc492ec19ef4bd408c370078907b8` |
| seed 29 C1-best1 | `sha256:9f7974619596b994ecce0f58517ebe0d9f40215c98b56748966b975c96c0a25f` |
| seed 43 B2b | `sha256:e76540925b835a62524e5ded1427f3d6196a2088070fef7881e2c8a9bebf0344` |
| seed 43 C2-best1 | `sha256:4db42299596af95472c43b0c11f0ee7c671f0c2a16bee1c66b8d5793ca4cd551` |

Generation roots 19117, 19127, 19152, and 19162 were then detached from the
redundant array-wide no-op gate. Job 19117 began its four generation shards on
node02 at approximately 06:23 IST; the other three roots remained normally
queued behind the account GPU limit. Their generation gates and every later
dependency remain unchanged. This overlaps evaluation of completed artifacts
with on-policy training and changes wall-clock scheduling only.

### Full offline-evaluation overlap

The same gate was then applied to the other four completed offline adapters.
Each destination path on node02 was confirmed absent before copying, and each
inference-artifact identity matched its source on node01 afterward:

| Chain | Adapter identity | Jobs pinned to node02 |
|---|---|---|
| seed 29 B2b | `sha256:8f04eee80641d7c24047b4f4b1e6b311ecc27c59ec11e97948ecc8dc10e82942` | 19122--19126 |
| seed 29 C2-best1 | `sha256:99dff31d8f6b5984bf958c36dd925e8d9208e476c3fe49e242aaebd42fe95a9c` | 19132--19136 |
| seed 43 B2a | `sha256:1ddf8e26b307c37275beaf4bfb4f5113f9a416c981ba0cc0e8a56f36806f7df4` | 19147--19151 |
| seed 43 C1-best1 | `sha256:8e324dee3368c0d404ba2f2f550d728c87a05579f2874b6a870513c90129e435` | 19157--19161 |

Generation roots 19122, 19132, 19147, and 19157 were detached from the same
redundant array-wide no-op gate. Node02 can consequently process all eight
finished offline-method chains while node01 trains the four on-policy methods.
After training, node01 retains seed-29 D2 and seed-43 D1, while node02 retains
the held seed-29 D1 and seed-43 D2 chains. The expected post-training evaluation
tail is therefore approximately two chains per node rather than six.

### Automated release gates

Three CPU-only, dependency-gated jobs remove the need for an operator to race
the end of training or evaluation. They are fail-closed: a copy, source lookup,
byte-hash comparison, or Slurm release failure exits nonzero and leaves the
downstream job held.

- job 19262 runs on node02 after final-preflight task `19111_13`, stages the
  seed-29 D1 final adapter from node01, compares the exact official
  `model_artifact_identity` used by generation, and only then releases
  generation root 19137;
- job 19263 does the same after `19113_14` for seed-43 D2 and releases root
  19172;
- job 19261 runs on node01 after the ten evaluation jobs assigned to node02
  (`19121`, `19126`, `19131`, `19136`, `19141`, `19151`, `19156`, `19161`,
  `19166`, and `19176`). It copies each exact corrected-v2 evaluation JSON to
  the declared node01 path, compares its byte SHA-256, and only after all ten
  pass releases held aggregate-analysis job 19177.

The two on-policy staging jobs depend on the original final-preflight tasks,
not on intermediate checkpoint presence. The repatriation job depends on final
evaluation JSON producers, not merely generation or scoring gates. All three
jobs request no GPU and change only storage availability and held-job state.
Pending jobs 19259 and 19260 were canceled before eligibility and superseded
by 19262 and 19263: the first revision compared the two required LoRA files,
whereas the replacement deliberately calls the repository's serving-time
identity function so any additional inference-relevant config, tokenizer,
index, or weight file is also bound.

### Node02 judge-cache completion

The first B2a score workers (array 19119) discovered that node02's pinned
`Qwen/Qwen3-32B-FP8` snapshot initially contained two complete weight blobs and
five incomplete blobs, so SGLang's normal Hugging Face loader completed the
missing cache before opening its health ports. The shared cache locks produced
one incomplete file per distinct missing blob rather than duplicate candidate
scores. At approximately 07:22 IST the cache had zero incomplete files, all
seven snapshot weight links, 32 GiB of blobs, and HTTP 200 from all four worker
health endpoints. No scoring worker was requeued and no score request ran
before its server health gate. The frozen model revision remained
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.

After health, concrete B2a score task 19266 completed and atomically wrote 21
of its 414 assigned prompt files, then SGLang exited on a 194 MiB activation
allocation when the `0.90` static-memory reservation left only 44.56 MiB free.
The client surfaced the disconnect and task 19266 exited 1; no partial prompt
JSON was committed. Resumable replacement job 19271 was submitted for exactly
score shard 2 of 4 with the same generation/config/prompts/judge and
deterministic inference, `JUDGE_MEM_FRACTION_STATIC=0.85`, score concurrency 4,
and PyTorch's expandable-segments allocator guard. It validated and reused the
21 complete files, scored the remainder, and completed successfully at 08:15
IST after 25 minutes 49 seconds.

Original B2a shard 3 (concrete job 19119) subsequently hit the same failure
class after 195 complete prompt files: a 360 MiB allocation failed with only
318.56 MiB free while the process used 44.07 GiB. It exited 1. Resumable job
19288 was therefore submitted with the identical scientific request and the
same `0.85`/expandable-segments repair, explicitly bound to score shard 3.
Slurm allocated this job at 08:13 IST but canceled its batch step after one
second with `RaisedSignal:53(Real-time_signal_19)` and no batch stdout or
stderr. The script body never began and no score record was written. `sacct`
retained `AssocGrpGRES` in its reason column, but completed jobs retain the
same earlier pending reason, so that field does not identify the signal's root
cause. The immediate failure is therefore an unresolved scheduler-launch
event, not a scientific or model failure.

Fresh replacement 19295 was submitted with an explicit node02 scratch working
directory and scratch stdout/stderr paths. It retains the identical request,
shard index 3, `0.85` memory fraction, expandable-segments guard, and resumable
reuse of the 195 valid atomic records. It entered `RUNNING` at 08:27:54 IST,
and the recovery watcher then released all temporary scheduling holds. Job
19295 completed 414/414 shard records with exit 0 at 08:43:15 IST after 15
minutes 21 seconds. Because score gate 19120 had already
entered Slurm's irreversible `DependencyNeverSatisfied` state after 19288,
replacement global gate 19297 was created after 19295, downstream evaluation
19121 was rebound to 19297, and the never-runnable zero-runtime gate 19120 was
canceled. Gate 19297 still performs the original global 1,658-prompt
completeness/content validation, so downstream evaluation cannot run on the
partial shard. The original generation gate 19118 was independently rechecked
as `COMPLETED 0:0` before the replacements were submitted without a historical
Slurm dependency (Slurm rejected adding a new dependency on that
already-finished old job).

Replacement gate 19297 ran immediately after the repair and completed with
exit 0 at 08:43:26 IST. Its frozen status checker reported exactly
`{"complete": true, "pending": 0, "total": 1658}` and the final file-count
assertion also passed. Corrected evaluation job 19121 is consequently eligible
for its ordinary GPU allocation; it no longer carries any dependency on the
failed or canceled jobs.

At that point Slurm's fixed-priority queue continued filling freed GPUs with
older generation-array elements. To prevent the completed evaluation stage
from starving behind all remaining generations, only pending eligible node02
GPU jobs were temporarily held; four running jobs were untouched. Evaluation
19121 took the next GPU at 08:52:47 IST, and an automatic watcher immediately
released every temporary hold. This stage-balancing change altered neither an
executable nor a scientific parameter and left node02 fully utilized.

Evaluation 19121 completed with exit 0 at 09:09:21 IST after 16 minutes 34
seconds. It embedded all 21,264 initially missing vectors, atomically wrote the
13,276,716-byte corrected-v2 JSON, and produced SHA-256
`8cd184a5aa40e7a9f847e0a1394d9d1f8db2cbb8da42c93127c3910c6d526e90`.
Its seed-29 B2a feasibility/soundness means are 3.360223/3.153046; this is an
individual checkpoint result, not the across-seed conclusion.

Afterward, the eligible queue again contained four older generation arrays
ahead of the ready B2b/C1 score work. Pending generation jobs alone were held
until B2b score tasks 19274_2 and 19274_3 occupied the two newly freed slots at
09:08:18 and 09:09:21 IST; the zero-safe watcher then released all holds.
Running work was never suspended, node02 remained fully occupied, and all
model, judge, sample, and decoding settings were unchanged.

The same stage-balancing rule was applied once more when C2 generation tasks
1 and 2 approached completion: pending generation alone was held until C1
score task 19275_0 started at 09:18:43 IST, then immediately released. C2 task
3 took the following freed GPU at 09:20:11. The resulting allocation—three
judge workers and one generator—kept both stages advancing without changing
or pausing any running job.

As the two B2b workers neared completion, pending seed-43 generations and C1
score tasks 2--3 were held long enough to reserve the two released GPUs for
B2b evaluation and C1 score task 1. B2b score tasks 19274_2/3 completed with
exit 0 at 09:35:35/09:36:31; global gate 19125 then reported exactly
`{"complete": true, "pending": 0, "total": 1658}` and exited 0 in three
seconds. Evaluator 19126 started at 09:36:34, after which all temporary holds
released. The allocation became one evaluator, one generator, and two C1
judge workers. Pending seed-43 generation roots were then briefly held again
so the evaluator's next freed GPU would start a third C1 judge shard rather
than another older generation shard. Evaluator 19126 completed with exit 0 at
09:41:48 after 5 minutes 14 seconds, C1 task 19275_2 started at that exact
timestamp, and the zero-safe watcher released all four temporary holds. Only
the three intentional artifact/analysis safety holds remained.

The B2b evaluator consumed 14,635 cached embeddings and rebuilt the 6,629
missing embeddings. Its 13,284,916-byte corrected-v2 artifact has SHA-256
`b03bbc6c32e83ef32f2d84319f3aa09414c8ab2f599843479196e944b3d78a65`.
The artifact declares all 1,658 prompts, four student and four teacher samples
per prompt (from four and sixteen source samples respectively), and all eight
frozen threshold-sensitivity entries. Seed-29 B2b feasibility/soundness are
3.409530/3.197527. This is a single-checkpoint result, not an across-seed
conclusion.

When C1 shard 0 finished at 09:47:50, Slurm correctly returned one GPU to
seed-43 generation. The still-pending seed-43 generation elements were then
held once more so that C1's fourth and final score shard could take the next
freed GPU. C2 generation shard 19132_3 completed with exit 0 at 09:56:53,
bringing the fail-closed generation count to exactly 1,658; gate 19133 exited
0 two seconds later. C1 score task 19275_3 started on that freed GPU at
09:56:53, and the watcher released every temporary hold. Running seed-43 B2a
generation task 19147_0 was never paused. The node therefore continued with
three judge workers and one generator, while C2 scoring became eligible for a
subsequent GPU.

C1 score task 19275_1 completed with exit 0 at 10:04:19. Pending seed-43
generation elements had been temporarily held so the newly eligible C2 score
array would not sit behind every older generation root. C2 score task 19276_0
started at the same timestamp, and the watcher released all temporary holds;
the only user holds again became the two cross-node final-adapter safeguards
and the final aggregate-analysis safeguard.

By 10:04 IST all four on-policy runs had written checkpoint 75/125. Each of
the four checkpoint directories was then checked for non-empty adapter config,
132,187,888-byte adapter weights, 264,673,227-byte optimizer state, scheduler
state, RNG state, and trainer state. Their adapter hashes are distinct across
method and seed. Training continues toward checkpoint 125; no intermediate
checkpoint is eligible for downstream evaluation.

| checkpoint-75 adapter | SHA-256 |
|---|---|
| seed 29 D1 | `2d4dfc9e0e9d967a53f89b169e4d309b4b84c87279cb3d0957184e3542212d63` |
| seed 29 D2 | `e3b729ad9d3c21579918828cd0ca9250e86331e36596c0d8060672a9faba6325` |
| seed 43 D1 | `a121805380a3297d3ddffecc4c7ee453aba9569cc7790a4bb29979618558c308` |
| seed 43 D2 | `599946a9ee03f3741dc1c807f7e0d240d92b19e48c6390817658e5715fe8ec45` |

C1 score task 19275_2 completed with exit 0 at 10:10:18, and Slurm returned
that GPU to seed-43 B2a generation task 19147_1. B2a task 19147_0 completed
with exit 0 at 10:11:37, and task 19147_2 started immediately. Before C1's
last score worker exited, only pending generation tasks and pending C2 score
tasks 1--3 were held so its evaluator could claim the next freed GPU. C1 task
19275_3 completed with exit 0 at 10:26:44; gate 19130 validated the exact
1,658-record score set and exited 0 in two seconds; evaluator 19131 started at
10:26:47. The watcher then released every temporary hold. The stable running
mix became one evaluator, one C2 judge worker, and two seed-43 generators.

Evaluator 19131 completed with exit 0 at 10:34:13 after 7 minutes 26 seconds.
It reused 14,632 cached embeddings, rebuilt 6,632, and atomically wrote a
13,274,613-byte artifact with SHA-256
`ac2db25f89ec3b6bd8c58aac4ac71fca479fcad87da771b07d39c082e7a1ad1a`.
The artifact declares all 1,658 prompts, matched K=4 student/teacher samples,
and all eight threshold-sensitivity entries. Seed-29 C1
feasibility/soundness are 4.240501/4.659228. A checksum-matched local raw copy
passed the same structural audit.
The sorted prompt-ID fingerprint for seed-29 B2a, B2b, and C1 is identical:
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`.
Their interim comparisons are therefore exactly prompt-paired rather than
merely equal-sized.

C2 score task 19276_0 completed with exit 0 at 10:33:08 and task 1 started at
the same timestamp. Pending generation alone was briefly held so the next
available lane would start C2 score task 2. Task 2 started at 10:33:44, and
the watcher released every temporary hold, leaving two judges and two
generators after the C1 evaluator exited.

Seed-43 B2a generation task 19147_3 completed with exit 0 at 10:57:39,
bringing the generation set to exactly 1,658 prompts; gate 19148 exited 0 two
seconds later. C2 score task 19276_3 started on that lane at 10:57:39. After
the prior generation hold released, pending generation was re-held only after
a 20-second handoff guard so the two watchers could not race. Seed-43 B2b
generation task 19152_0 completed with exit 0 at 10:59:19, and B2a score task
19279_0 started one second later. All temporary holds then released. No
running task was suspended, and every scientific parameter remained frozen.

Because the same `0.90` static-memory setting was frozen into the eleven
remaining score arrays before the node02 failure was understood, leaving
those arrays in place would knowingly repeat the same avoidable OOM risk.
They were canceled while pending (apart from B2b seed 29 shard 0, which had
started but had not completed) and replaced with otherwise identical arrays
using `JUDGE_MEM_FRACTION_STATIC=0.85` and the allocator guard. Existing
atomic per-prompt files are validated and reused. Each original score gate was
rebound to its replacement array before the old array was canceled:

| method | seed | node | old score array | replacement | score gate |
|---|---:|---|---:|---:|---:|
| B2b | 29 | node02 | 19124 | 19274 | 19125 |
| C1-best1 | 29 | node02 | 19129 | 19275 | 19130 |
| C2-best1 | 29 | node02 | 19134 | 19276 | 19135 |
| D1 | 29 | node02 | 19139 | 19277 | 19140 |
| D2 | 29 | node01 | 19144 | 19278 | 19145 |
| B2a | 43 | node02 | 19149 | 19279 | 19150 |
| B2b | 43 | node02 | 19154 | 19280 | 19155 |
| C1-best1 | 43 | node02 | 19159 | 19281 | 19160 |
| C2-best1 | 43 | node02 | 19164 | 19282 | 19165 |
| D1 | 43 | node01 | 19169 | 19284 | 19170 |
| D2 | 43 | node02 | 19174 | 19285 | 19175 |

Scheduler inspection confirmed every unaffected gate depends on the
corresponding new array wildcard and all eleven old roots are `CANCELLED`.
Completed repair 19271 and fresh repair 19295 cover B2a seed-29's failed
shards; replacement global gate 19297 is fail-closed behind 19295 and validates
all four shards. Only pending generation/scoring work was temporarily held to
give repair 19295 the next available node02 GPU; those temporary holds release
automatically as soon as the repair starts. The intentional final-adapter and
aggregate-analysis holds described above remain separate and unchanged.

C2 score tasks 19276_0--3 completed with exit 0, with the final shard ending
at 11:27:34 IST after 29 minutes 55 seconds. Gate 19135 then validated exactly
1,658 complete prompt score sets and exited 0 in two seconds. Only pending
generation and seed-43 B2a score tasks were held before that transition;
evaluator 19136 started at 11:27:37 and every temporary hold was released.
Three already-eligible generation tasks filled the other node02 GPUs, so the
node remained fully utilized. A second zero-safe handoff held only still-
pending generation tasks until B2a score task 19279_1 started after evaluator
completion, then released them; running jobs were never suspended.

Evaluator 19136 completed with exit 0 at 11:34:27 after 6 minutes 50 seconds.
Its 13,273,760-byte corrected-v2 artifact has SHA-256
`41b697004e419df85d9f1221816e8bb4f263d3619cbd1380120bfa460dcd6147`.
It declares schema version 1, all 1,658 prompts, matched K=4 student/teacher
samples from four/sixteen source samples, and all eight frozen threshold
entries. Seed-29 C2 feasibility/soundness are 4.238691/4.655458; descriptive
strict-threshold qualified yield is 1.741255 and teacher-mode recall is
0.134449. The checksum-matched local copy passed the same structural checks.
Its sorted prompt-ID fingerprint is
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`,
identical to seed-29 B2a, B2b, and C1. Relative to seed-matched C1, C2 is
-0.001809 feasibility, -0.003770 soundness, and -0.437274 qualified yield,
with identical teacher-mode recall. The quality tie and lower embedding-
defined breadth reproduce the seed-17 direction, but the breadth result
remains quarantined until human semantic-boundary calibration.

All four on-policy jobs subsequently produced complete checkpoint-100
recovery states. Each directory has `global_step=100`, exactly 100 finite loss
and gradient records, a rank-16 adapter with 504 tensors, 132,187,888-byte
adapter weights, 264,673,227-byte optimizer state, and non-empty scheduler,
RNG, and trainer state. The distinct adapter identities are:

| checkpoint-100 adapter | SHA-256 | step-100 loss | step-100 gradient norm |
|---|---|---:|---:|
| seed 29 D1 | `5c25c4fdc1e02a4273a726b5686e0973ed1b0758ad3f2cd438a477f1e52ba146` | 2.2958 | 1.803092 |
| seed 29 D2 | `86526cf5f02c12c913504a0465a7f204b5ee2bcb8bd43034e62b63acfe7a239b` | 2.7600 | 2.715563 |
| seed 43 D1 | `d231265e785b93024c97416d971e42c3686644999ede854dbfd7679a39e60257` | 2.5718 | 1.929148 |
| seed 43 D2 | `5425657907b82b1ec0e1f97fbdd9d6f18daae6f600d9eed15dfa3a781f09b2e3` | 2.6483 | 2.529509 |

These are recovery and optimization-integrity diagnostics only. Training
continues to step 125, and no checkpoint-100 adapter is eligible for held-out
evaluation.

Seed-43 B2b's final generation shard completed with exit 0 at 11:51:08 after
23 minutes 27 seconds. Gate 19153 validated the exact 1,658-prompt generation
set and exited 0 in seven seconds. Pending C1/C2 generation tasks alone had
been held so B2a score task 19279_2 could take the released lane; it started
at exactly 11:51:08, and the zero-safe watchers released every temporary
hold. The node returned to two judge workers and two generators. When B2a
score task 19279_1 approached completion, only still-pending C1/C2 generation
tasks were held again. Task 19279_1 completed with exit 0 at 12:01:38 after
27 minutes 11 seconds, task 19279_3 started at the same instant, and the
watcher released all temporary holds. No running work was paused or changed.

B2a score task 19279_2 completed with exit 0 at 12:19:27 after 28 minutes 19
seconds. Pending C2 generation alone had been held so ready B2b score task
19280_0 could preserve the two-judge/two-generator balance; it started on that
same lane and the watcher released the hold. Before B2a's final score task
ended, only pending C2 generation and B2b score tasks were held to reserve the
next lane for the now-complete B2a chain. Task 19279_3 completed with exit 0
at 12:28:50 after 27 minutes 12 seconds, gate 19150 validated exactly 1,658
score sets and exited 0 in two seconds, and evaluator 19151 started at
12:28:53. The temporary holds then released.

Evaluator 19151 completed with exit 0 at 12:34:07 after 5 minutes 14 seconds.
It reused 14,665 cached embeddings, rebuilt 6,599, and atomically wrote the
13,283,421-byte corrected-v2 artifact with SHA-256
`5cc9714fdc190b023a3c7634a3c3d3474bf68c4935f0877b065b285aabc82f81`.
The artifact declares schema version 1, all 1,658 prompts, matched K=4
student/teacher samples from four/sixteen source samples, and all eight frozen
threshold entries. Seed-43 B2a feasibility/soundness are 3.397919/3.180036;
qualified yield is 0.468637 and teacher-mode recall is 0.025533. The local
copy is checksum-identical and has the common sorted prompt-ID fingerprint
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`.
B2a is now complete over checkpoint seeds 17/29/43; its fixed-A0-conditioned
across-seed feasibility and soundness deltas are -0.832328 and -1.436369,
with seed SDs 0.021677 and 0.013511 respectively. C2 generation was then held
only until B2b score task 19280_1 took the evaluator's released lane at
12:34:08; the watcher released immediately, with running work untouched.

Seed-43 B2b score tasks 19280_0--3 all completed with exit 0. Their exact
end times were 12:47:52, 13:01:19, 13:16:07, and 13:28:37 IST. Gate 19155
then validated exactly 1,658 complete score sets and exited 0 in three
seconds. Evaluator 19156 started at 13:28:40; a zero-safe watcher released
the three temporarily held C1 score workers as soon as that evaluator was
running. Those workers returned to normal Slurm resource-pending state; no
running task was paused.

Evaluator 19156 completed with exit 0 at 13:33:57 after 5 minutes 17 seconds.
It reused 14,694 cached embeddings, built the remaining 6,570, and atomically
wrote a 13,282,765-byte corrected-v2 artifact with SHA-256
`6989ed29df85943fa2e8a5ea542f84d969c20722cc99d2b878114eb9cfe7427d`.
The artifact declares schema version 1, all 1,658 prompts, matched K=4
student/teacher samples, and all eight threshold-sensitivity entries. Its
sorted prompt-ID fingerprint is the common
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`.
Seed-43 B2b feasibility/soundness are 3.412545/3.195416; qualified yield is
0.488540, teacher-mode recall is 0.026940, and the length-stop rate is zero.
The checksum-matched local copy passed the same structural audit.

B2b is therefore complete over checkpoint seeds 17/29/43. Its across-seed
feasibility/soundness means are 3.406815/3.189686. Relative to seed-matched
B2a, its mean changes are only +0.021562 feasibility (seed SD 0.025008;
descriptive df=2 t interval [-0.040562, 0.083686]) and +0.023522 soundness
(seed SD 0.018301; interval [-0.021940, 0.068984]). Best-of-eight judge
selection therefore does not repair the hard single-output SeqKD failure.

Meanwhile, the first two seed-43 C2 generation shards completed with exit 0
at 13:22:01 and 13:22:00. Shards 2 and 3 started immediately on the two freed
node02 lanes. Seed-43 C1 score shard 0 continued concurrently; its remaining
three score workers were unheld after the B2b evaluator handoff and await
the account's currently saturated GPU allocation.

Seed-29 D1 and D2 reached step 125 and wrote their final metadata at 13:36:26
and 13:37:53. Their final adapter hashes match their checkpoint-125 hashes:
`6b69301f105c915d5845973394075bd88111f2b7c4dd523a055ec56b7043069f`
for D1 and
`42580b90dd9a679e515016f607ac3fc6ddaed45a76e7c430777be1aa2f42dd57`
for D2. Both trainer states declare global step 125, epoch 1.0, exactly 125
finite loss/gradient records, and the frozen 1,000-exposure contract. D1's
final loss/gradient are 2.4447/1.880429; D2's are 2.5787/2.644467.

The seed-29 no-op resume array exposed one deployment-staleness edge. Concrete
task 19206, the already-complete B2b preflight, exited 1 before any scientific
work because its old spooled batch script did not contain the later
`PYTHONPATH=${repo_dir}/src...` safeguard; another concurrent no-op task was
reinstalling the shared editable package, so `check_training_status.py`
briefly could not import `novelty_distill`. B2b's previously validated adapter
and completed evaluation were not read or changed by this failed no-op. The
current repository already contains the source-import safeguard and its
focused concurrency-contract test passes. Because the still-spooled seed-43
resume array predates that fix as well, its pending task throttle was reduced
to one before release. Those completion preflights will therefore serialize
instead of entering the same transient uninstall window; this is a scheduler
mitigation only and does not change training.

Stage job 19262 then checksum-staged the final seed-29 D1 adapter from node01
to node02. Source and destination adapter files are both 132,187,888 bytes and
have the same SHA-256
`6b69301f105c915d5845973394075bd88111f2b7c4dd523a055ec56b7043069f`.
Only after the complete tree identity gate passed did it release generation
array 19137. Seed-29 D2 generation shards 0 and 1 simultaneously began on
the two node01 GPUs freed by completed training; the other two shards remain
normally resource-pending until the seed-43 training lanes finish.

Seed-43 D2 and D1 then completed step 125 at 13:40:30 and 13:46:56. Their
final adapters again match their checkpoint-125 adapters byte for byte. D2's
SHA-256 is
`09e4cee5f80e88ce1f3404c5e19042c6c8b1d96b8da3cc08384aec347631b6c4`;
its final loss/gradient are 2.5208/2.803363. D1's SHA-256 is
`8889c3873d1b22a5642909f7395d220f6de41ea13a726f5f20514d6d9d36f28d`;
its final loss/gradient are 2.1940/1.826337. Both trainer states declare step
125, epoch 1.0, exactly 125 finite loss/gradient records, and 1,000 optimizer
example exposures. All four new on-policy training replications are now
complete; none of their checkpoint-75 or checkpoint-100 recovery states was
used for evaluation.

A direct fail-closed audit of the four final safetensors headers found exactly
504 non-empty F32 tensors in each adapter. Every final file matches its
checkpoint-125 file byte for byte, all four adapter hashes are distinct, and
the ordered 1,000-example fingerprint is identical across runs:
`190e3ea7cb48a13dbd5a142f3aa85ff5f355fd69ffd89bf8ee75a47c9d3aa7dc`.

The serialized seed-43 completion-preflight array ran all six declared tasks
one at a time from 13:46:56 through 13:47:12. Every task exited 0, including
the B2b task that exercises the deployment-staleness mitigation. Stage job
19263 then copied seed-43 D2 from node01 to node02. Both adapter files are
132,187,888 bytes with SHA-256
`09e4cee5f80e88ce1f3404c5e19042c6c8b1d96b8da3cc08384aec347631b6c4`;
only after the complete-tree identity matched did the job release generation
array 19172. Seed-29 D2 generation expanded to all four node01 lanes as the
last training GPUs became available. Seed-43 D1 remains normally pending on
node01 behind those already-running generation shards, while the two staged
node02 methods remain resource-pending behind the active offline evaluation.

When seed-43 C1 score shard 0 completed at 13:46:10, seed-29 D1 generation
shard 0 won the newly available node02 lane before C1 score shard 2. That
running generation was left untouched. Only its three still-pending sibling
shards were temporarily held so C1's remaining score shards can take the
next freed GPUs and reach their evaluator gate. A trap-protected watcher will
release those three siblings as soon as evaluator 19161 is running.

Seed-43 C2 generation shards 2 and 3 completed with exit 0 at 13:59:25 and
13:59:49. Gate 19163 then validated exactly 1,658 complete prompts with zero
pending and exited 0 in two seconds. The two freed GPUs started the remaining
seed-43 C1 score workers at 13:59:25 and 13:59:49. When C1 score shard 1
completed at 14:02:36, C2 score shard 0 started on the same lane. This kept
node02 at one on-policy generator, two C1 judges, and one C2 judge.

The final C1 score workers both completed with exit 0 at 14:28:34 after
29:09 and 28:45. Gate 19160 validated the exact 1,658-record score set in two
seconds, and evaluator 19161 started at 14:28:37. C2 score shard 2 was then
released and began at 14:28:56. C2 shard 0 completed at 14:31:16 after
28:40, allowing C2 shard 3 to start; all remaining pending on-policy
generation holds were released only after C2 score shards 2 and 3 were both
confirmed running.

Evaluator 19161 completed with exit 0 at 14:35:33 after 6 minutes 56 seconds.
It reused 14,633 cached embeddings, built 6,631, and atomically wrote a
13,274,447-byte corrected-v2 artifact with SHA-256
`20935d1bd974c55e54c18e527865e253c7bf905a4486526ad02df2ef4bc444f8`.
The artifact has all 1,658 prompts, matched K=4 samples, all eight threshold
entries, and the common prompt fingerprint
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`.
Seed-43 C1 feasibility/soundness are 4.254976/4.670989; qualified yield is
2.167069, teacher-mode recall is 0.135856, and one of 6,632 outputs ended by
length limit. The checksum-matched local copy passed the same structural
audit.

C1 is now complete over checkpoint seeds 17/29/43. Its across-seed
feasibility/soundness means are 4.244672/4.662545. Relative to seed-matched
B2b, the mean changes are +0.837857 feasibility (seed SD 0.006070; descriptive
df=2 t interval [0.822779, 0.852934]) and +1.472859 soundness (seed SD
0.010079; interval [1.447822, 1.497896]). The large KL recovery from hard
single-output KD is stable across the three trained checkpoints. The
descriptive C1-minus-fixed-A0 means are +0.027091/+0.060012, but those reuse
one A0 generation realization and exclude its sampling uncertainty, so they
support competitiveness rather than a general superiority claim.

On node01, all four seed-29 D2 generation shards completed with exit 0 by
14:25:19. Gate 19143 validated exactly 1,658 prompts with zero pending and
exited 0 two seconds later. Each freed lane immediately began the
corresponding seed-43 D1 generation shard, keeping all four node01 GPUs active
while seed-29 D2 scoring waits for those already-running generations.

Seed-43 C2 score shards 1--3 completed with exit 0 at 14:54:42, 14:57:56,
and 14:59:52 after 29:53, 29:00, and 28:36. Gate 19165 validated the complete
1,658-record score set in four seconds, and evaluator 19166 started at
14:59:56. Only then did the trap-protected watcher release the pending
on-policy node02 generation elements. Evaluator 19166 completed with exit 0
at 15:06:48 after 6 minutes 52 seconds and atomically wrote a 13,271,349-byte
artifact with SHA-256
`059835a112662534002f2a7d04c4ac31ab51f2f3da29239b7b43c2ff556b7e19`.
The checksum-matched local copy has 1,658 prompt entries, K=4, all eight
threshold views, primary threshold 0.94, and the common prompt fingerprint
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`.
Seed-43 C2 feasibility/soundness are 4.237636/4.652593; qualified yield is
1.756333, teacher-mode recall is 0.141335, and there were zero length stops.

C2 is now complete over checkpoint seeds 17/29/43. Its across-seed
feasibility/soundness means are 4.238741/4.654604. Relative to seed-matched
C1, the mean changes are -0.005931 feasibility (seed SD 0.010007; descriptive
df=2 t interval [-0.030789, 0.018927]) and -0.007941 soundness (seed SD
0.009115; interval [-0.030584, 0.014702]). Static forward and reverse KL are
therefore quality ties. The mean strict-threshold qualified-yield change is
-0.434861 (seed SD 0.023014; interval [-0.492032, -0.377691]), a stable but
still-quarantined embedding diagnostic pending the two-human boundary study.

On node01, the last seed-43 D1 generation shard completed at 15:04:21 after
39:02 and gate 19168 validated all 1,658 prompts in two seconds. D2 seed-29
score shard 3 immediately took the freed lane; all four D2 score shards are
now running, with D1 seed-43 scoring dependency-ready behind them.

Seed-29 D2 score shards 0--3 completed with exit 0 at 15:20:31, 15:23:36,
15:24:58, and 15:30:47; runtimes were 26:38, 26:33, 26:16, and 26:26. The
first three freed lanes immediately started seed-43 D1 score workers. Only
pending D1 score task 3 was trap-safely held after those first three D2
shards had completed, reserving one lane for the D2 global transition. Gate
19145 validated exactly 1,658 records in four seconds and evaluator 19146
started at 15:30:51. The held D1 task was released only after the evaluator
was confirmed running.

Evaluator 19146 completed with exit 0 at 15:37:53 after 7 minutes 2 seconds.
The checksum-matched local copy is 13,273,929 bytes with SHA-256
`2158a02b3017e9f17385a57683f2f59b46a52c09afc15930f2c6e9a0be59bd7e`.
It has all 1,658 prompt entries, K=4, all eight threshold views, and the common
prompt fingerprint
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`.
Seed-29 D2 feasibility/soundness are 4.226176/4.638420; qualified yield is
1.731001, teacher-mode recall is 0.124849, and two outputs ended by length
limit. Relative to seed-matched C2, the changes are -0.012515 feasibility,
-0.017039 soundness, -0.010253 qualified yield, and -0.009600 recall. D1
seed-43 score task 3 started immediately when the evaluator freed its lane.

Seed-43 D1 score shards 0--3 completed with exit 0 at 15:47:12, 15:50:14,
15:51:19, and 16:04:11; runtimes were 26:41, 26:38, 26:21, and 26:18. Gate
19170 validated the exact 1,658-record score set in three seconds, and
evaluator 19171 started at 16:04:14. It completed with exit 0 at 16:11:16
after 7 minutes 2 seconds. The checksum-matched local copy is 13,272,191
bytes with SHA-256
`7bb7806cbd50dc88a1ccac98b381404b7f6d453a5f82f348ea919fade51331e3`.
It has 1,658 prompt entries, K=4, all eight threshold views, and the common
prompt fingerprint
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`.
Seed-43 D1 feasibility/soundness are 4.251960/4.675513; qualified yield is
2.138118, teacher-mode recall is 0.108665, and one output ended by length
limit. Relative to seed-matched C1, the changes are -0.003016 feasibility,
+0.004524 soundness, -0.028951 qualified yield, and -0.027191 recall.

On node02, seed-29 D1 generation shards 1--3 completed with exit 0 at
15:14:39, 15:38:39, and 15:39:26. Gate 19138 validated exactly 1,658 prompts
in two seconds. Seed-43 D2 generation took the freed lanes; as those shards
completed, D1 score shards 0--3 started at 15:45:21, 15:53:24, 16:13:56,
and 16:16:40. They completed with exit 0 at 16:13:56, 16:22:00, 16:42:30,
and 16:46:32 after 28:35, 28:36, 28:34, and 29:52.

After the first two D1 score shards had completed and two D2 score shards
were running, only still-pending D2 score tasks 2--3 were held to reserve the
D1 evaluator transition. Gate 19140 validated the exact 1,658-record D1 set
in two seconds, and evaluator 19141 started at 16:46:35. The held D2 tasks
were released after the evaluator was confirmed running; task 2 started at
16:46:39 and task 3 at 16:47:08, while all already-running work was untouched.

Evaluator 19141 completed with exit 0 at 16:53:35 after exactly 7 minutes.
The checksum-matched local copy is 13,277,773 bytes with SHA-256
`95c3bc81d007555aca7ed1b232e23d47a057c549539fac5482e8b26c053f0f4b`.
It has 1,658 prompt entries, K=4, all eight threshold views, and the common
prompt fingerprint
`b249121a0312fc9060de57d1d2a95852e4279fc0f075fe426503582d7b14b20c`.
Seed-29 D1 feasibility/soundness are 4.231604/4.663148; qualified yield is
2.167672, teacher-mode recall is 0.118114, and two outputs ended by length
limit.

D1 is now complete over seeds 17/29/43. Its across-seed
feasibility/soundness means are 4.239043/4.665661. Relative to seed-matched
C1, mean changes are -0.005629 feasibility (seed SD 0.002994; descriptive
df=2 t interval [-0.013067, 0.001809]) and +0.003116 soundness (seed SD
0.001939; interval [-0.001700, 0.007932]). On-policy forward KL is therefore
a quality tie with static forward KL. Qualified yield changes by -0.022115
(interval [-0.046520, 0.002290]); teacher-mode recall changes by -0.019971
(interval [-0.035505, -0.004436]), a stable but still-quarantined embedding
diagnostic pending two-human calibration.

## Invariants

The GPU count, node placement, time limits, repository commit, dataset, ordered
example IDs, model and teacher revisions, optimizer, learning rate, context
limits, objective, seed, exposure budget, checkpoint paths, prompts, decoding,
and downstream completeness checks were not changed. The update affects only
pending host-memory reservations, evaluator server static GPU-memory fraction,
and allocator fragmentation behavior; it has no intended statistical or
modeling effect. Resumed scoring is safe because completed files are accepted
only after their request identities and schemas validate.
