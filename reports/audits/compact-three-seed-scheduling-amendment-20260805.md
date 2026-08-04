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
- Pending idempotent final-preflight tasks for seeds 29 and 43 were reduced to
  32,000 MiB because they either validate an already-complete checkpoint or
  resume the same bounded trainer under the checkpoint lock.
- The pending B2a NoveltyBench job was reduced from 96 GiB to 64,000 MiB. The
  completed matched A0 NoveltyBench job used about 25.2 GiB maximum RSS.

Slurm splits individually updated array elements into new concrete job IDs;
their stable identities remain the original array/task pairs (`19111_*`,
`19112_*`, and `19113_*`). Dependencies refer to those stable array/task
identities and remain intact.

## Invariants

The GPU count, node placement, time limits, repository commit, dataset, ordered
example IDs, model and teacher revisions, optimizer, learning rate, context
limits, objective, seed, exposure budget, checkpoint paths, and downstream
evaluation dependencies were not changed. The update affects only pending
host-memory reservations and therefore has no statistical or modeling effect.
