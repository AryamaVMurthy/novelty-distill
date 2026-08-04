# D2 on-policy reverse-KL K=4 extension

Status: complete and audited on 2026-08-04.

This extension adds the previously trained on-policy reverse-KL baseline
(`D2`) to the frozen compact K=4 study. It is a baseline comparison, not a
novelty claim. D2 used Qwen3-4B, 1,000 TOMATO training examples, 1,000 total
example exposures, 125 optimizer steps, seed 17, and student-generated
trajectories with reverse KL (`beta=1`, `lambda=1`).

The held-out evaluation used the same 1,658 TOMATO prompts, four generations
per prompt, frozen decoding, Qwen3-32B-FP8 judge, Qwen3-Embedding-4B semantic
representation, and teacher-mode construction as the completed six-method
study. The three D2 contrasts were declared before inspecting D2 held-out
outputs and form a separate Holm family; the original five contrasts remain
unchanged.

## Main result

D2 is a high-quality but narrow baseline. Its mean feasibility is 4.231/5,
soundness is 4.638/5, teacher-mode recall is 0.125, and viable semantic yield
is 1.737/4.

- Relative to on-policy forward KL (`D1`), feasibility and recall are not
  statistically distinguishable. Soundness is 0.020 lower, while viable
  yield is 0.486 lower (95% CI [-0.528, -0.441], Holm p=0.000300).
- Relative to off-policy reverse KL (`C2-best1`), feasibility and recall are
  not distinguishable. Soundness is 0.018 lower and viable yield is 0.058
  lower (95% CI [-0.097, -0.018], Holm p=0.00470).
- Relative to the practical forward-KL baseline (`C1-best1`), feasibility and
  recall are not distinguishable. Soundness is 0.019 lower and viable yield
  is 0.511 lower (95% CI [-0.554, -0.466], Holm p=0.000300).

The operational conclusion is that combining on-policy student trajectories
with reverse KL does not improve this baseline ladder. It retains near-ceiling
judge quality but produces fewer mutually distinct, quality-gated hypotheses.
`C1-best1` remains the main trained baseline; D2 is now the strongest
mode-seeking control in the current comparison.

## Files

- `findings.md`: all method levels, paired estimates, corrected p-values, and
  threshold-direction tables.
- `baseline-summary.csv`: compact seven-method metric table.
- `quality.svg` and `breadth.svg`: matched-K presentation plots.
- `paired-contrasts.json`: machine-readable paired analysis.
- `artifact-bundle.json`: hashes for the compact CSV and plots.
- `RUN_AUDIT.md`: training, generation, scoring, evaluation, job-recovery, and
  artifact audit.

The large prompt-level matrices remain on Turing node05 scratch:

`/scratch/node05/aryama.murthy/novelty-distill/evaluations/tomato/compact-k4-seed17-d2-extension/`
