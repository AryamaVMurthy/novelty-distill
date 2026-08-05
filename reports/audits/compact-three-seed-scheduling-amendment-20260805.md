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

## Invariants

The GPU count, node placement, time limits, repository commit, dataset, ordered
example IDs, model and teacher revisions, optimizer, learning rate, context
limits, objective, seed, exposure budget, checkpoint paths, and downstream
evaluation dependencies were not changed. The update affects only pending
host-memory reservations and therefore has no statistical or modeling effect.
