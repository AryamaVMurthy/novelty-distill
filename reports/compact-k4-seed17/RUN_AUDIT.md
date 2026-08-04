# Compact K=4 run audit

Producer commit: `c26a5dc1593b0d78fffe19637f7716574df29149`.

## Completeness

- Generation: 5 methods x 1,658 prompts x 4 candidates = 33,160 student
  candidates. All 20 Slurm array tasks exited 0.
- Judge scoring: 5 methods x 1,658 prompt files, each containing four records.
  All 20 Slurm array tasks exited 0.
- Generation validators: jobs `18656`, `18660`, `18664`, `18668`, `18672` all
  reported 1,658 complete and zero pending.
- Score validators: jobs `18658`, `18662`, `18666`, `18670`, `18674` all
  reported 1,658 complete and zero pending.
- Evaluations: jobs `18659`, `18663`, `18667`, `18671`, `18675` all exited 0.
- Paired analysis: job `18676` exited 0 and emitted 9,948 primary prompt rows
  plus 79,584 threshold rows.
- The cluster queue was empty at final audit.

## Frozen artifact identities

- `B1`: `sha256:cfb6e32ef2a0648b6d8e372f0d2325098faa352c8d3a6f4af1cfc55fa75cf280`
- `B2b`: `sha256:ee15f4037d9b0e92b79cb5b6291616f9d618a925d597e91938d7387e0ac6f2a8`
- `C1-best1`: `sha256:7fc74fcd56d0150faf987eb5234536cac649cd24882a4e35716ff2aedc1b50f8`
- `C2-best1`: `sha256:7080e929e3f72d2c58c4ef01b6afc86f8c190b351d79c8039fecff01eb6d9cf4`
- `D1`: `sha256:47a46794c06fdfdbe9b9109171de22c7588c00b5eb67d2e2ee2ed2f4d9ddd216`

Every generation manifest records the corresponding identity and the shared
K=4 generation config hash
`1b9e9cdff0df3636e611748a32d6d80fb0c4eeaec6b5a5489a56d0351ed6561d`.

## Evaluation provenance

- Held-out prompts: 1,658 TOMATO prompts.
- Selected teacher samples: 4 from a validated K=16 source.
- Selected student samples: 4. `A0` uses the deterministic first four from a
  validated K=16 source; trained methods were newly generated at K=4.
- Judge: `Qwen/Qwen3-32B-FP8`, revision
  `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.
- Shared teacher-score hash:
  `9227bb39afaecace106ea8f3a0b3ed87706145c7a3831f9bed0c11c22463de4b`.
- Training-reference targets: 8,000, hash
  `eea6b6e35f5d5e181740f656a5ec49a29ed40daa21ba3297ab26dc0ffc2fbd9f`.

## Compact artifact hashes

- `baseline-summary.csv`:
  `202ab5377b2c3a4f00b9a3931a211fc74a75fcd9d256a8457e075dd920e583f7`
- `quality.svg`:
  `d4c6a266aedd067e54ac94421f03ae00d6753ebc4dddcc89a13c6504fa324430`
- `breadth.svg`:
  `3dd28afbc95055087e57aa101bddde670e4d044200f8711ea96afb2c08ee4c58`

All three hashes were independently recomputed and match
`artifact-bundle.json`.

## Interpretation boundary

The metrics are operational frozen-judge and embedding outcomes. They do not
constitute human validation of scientific novelty. `A0` versus `C1-best1` is
descriptive in this report because it was not one of the five declared paired
contrasts.
