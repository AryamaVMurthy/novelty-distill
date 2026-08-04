# Corrected NoveltyBench smoke: A0, K=10, seed 17

This directory freezes the first portable smoke result produced after repairing
the repeated-generation bug in the official NoveltyBench adapter.

- Evaluation ID: `official-corrected-smoke-A0-k10-seed17-v2`
- Repository commit: `3e6ef48b33563103676f70f762e8d87f8d55bcd1`
- SLURM job: `19053` on Turing `node01`
- Dataset: official NoveltyBench curated split, first 2 prompts only
- Sampling: 10 calls per prompt with seeds 17 through 26, temperature 1.0,
  top-p 1.0
- Model: base Qwen3-4B revision
  `1cfa9a7208912126459214e8b04321603b3df60c`
- Result: Distinct@10 3.5 (SE 0.5), Utility@10 3.3189557613
  (SE 0.0521535442)
- Sampling diagnostic: 10/10 unique raw completions for both prompts; no
  duplicate-completion prompt

The result is a two-prompt implementation smoke, not an estimate suitable for
method comparison. The raw Inspect evaluation artifact and its machine-readable
summary are retained beside this file. The summary points to the artifact by a
relative path and was independently validated from the login-node filesystem
view.
