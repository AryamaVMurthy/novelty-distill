# Compact K=4 distillation baseline study

Status: complete and audited on 2026-08-04.

This is a baseline study, not a novelty claim. It compares the untouched
Qwen3-4B control (`A0`) with human SFT (`B1`), hard sequence KD (`B2b`),
off-policy forward KL (`C1-best1`), off-policy reverse KL (`C2-best1`), and
on-policy forward KL (`D1`). The teacher anchor (`A1`) supplies the reference
modes and is not treated as another student row.

All estimates use the same 1,658 held-out TOMATO prompts and four selected
student/teacher samples per prompt. The frozen local judge is
`Qwen/Qwen3-32B-FP8` at revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`. Confidence intervals use 10,000
paired bootstrap samples; sign-flip p-values use 10,000 Monte Carlo samples,
seed 17, with Holm correction within each metric.

## Main result

- `B1` is worse than `A0` on feasibility, soundness, teacher-mode recall, and
  viable semantic yield despite having more raw semantic clusters.
- `B2b` is worse than `B1` on feasibility, soundness, and viable yield; its
  small recall increase is not statistically distinguishable.
- `C1-best1` strongly recovers quality and viable breadth relative to `B2b`.
- `C2-best1` matches `C1-best1` on feasibility, soundness, and recall but loses
  0.453 viable modes per prompt, consistent with more mode-seeking behavior.
- `D1` does not improve feasibility, soundness, or viable yield over
  `C1-best1`, and its teacher-mode recall is 0.0164 lower.

The simplest baseline to carry forward is therefore `C1-best1`. `C2-best1`
is the useful low-breadth objective control, and `D1` is the on-policy control.

## Files

- `findings.md`: complete descriptive levels, paired estimates, confidence
  intervals, corrected p-values, and threshold curves.
- `baseline-summary.csv`: compact six-row metric table.
- `quality.svg` and `breadth.svg`: presentation plots.
- `paired-contrasts.json`: machine-readable paired analysis.
- `artifact-bundle.json`: hashes for the compact CSV and plots.
- `RUN_AUDIT.md`: job, provenance, validation, and artifact audit.

The prompt-level tables remain on Turing scratch because they are large:

`/scratch/aryama.murthy/novelty-distill/evaluations/tomato/compact-k4-seed17/`
