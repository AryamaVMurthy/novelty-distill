# Teacher sampling and evaluator calibration protocol

## Research question

For non-thinking Qwen3-14B teacher generation on TOMATO-Star, which frozen decoding condition
recovers the most semantic hypothesis modes without materially reducing judged quality or causing
systematic length truncation?

This is a calibration study. It does not test a trained student and its results are not headline
paper results.

## Fixed design

- Data: eight prompts selected as deterministic midpoints of equal prompt-character-length strata
  from the pinned 1,000-record TOMATO-Star training subset.
- Model: `Qwen/Qwen3-14B` at revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Per condition: eight independently seeded responses per prompt, seeds `17 + sample_index`.
- Inference: pinned SGLang 0.5.10 with batch-invariant deterministic mode and non-thinking Qwen chat
  templating.
- Shared controls: `top_k=20`, `min_p=0`, identical prompts, model revision, sample count, and seed
  policy.
- Conditions: temperatures 0.6, 0.7, 0.8, 1.0, and 1.2 at `top_p=0.8`; the previous
  `temperature=0.8, top_p=0.95` setting; and the official 0.7/0.8 setting with a 768-token rather
  than 512-token ceiling.

The exact matrix is `configs/generation/teacher_calibration.yaml`.

## Outcomes and analysis

Primary prompt-level outcomes:

1. instructed-embedding semantic cluster count at the predeclared cosine threshold 0.82;
2. quality-adjusted coverage;
3. mean fixed-judge quality;
4. within-prompt quality-score standard deviation;
5. fraction of responses ending at the token ceiling.

Sensitivity outcomes:

- raw and instructed Qwen3-Embedding-4B cluster counts at thresholds 0.70, 0.75, 0.80, 0.82,
  0.85, 0.90, and 0.95;
- pairwise cosine minimum, mean, and maximum;
- unique-text rate and response-token length;
- judge dimension distributions.

Compare every condition to the official Qwen non-thinking setting using prompt-paired differences,
paired bootstrap 95% confidence intervals, paired effect sizes, and Holm correction within each
metric family. With only eight prompts, emphasize intervals and failure cases; do not interpret a
p-value as decisive evidence.

## Predeclared hypotheses

- H1: increasing temperature raises semantic cluster count but eventually reduces mean quality.
- H2: the former `top_p=0.95` setting adds surface variation more readily than semantic modes.
- H3: the 768-token ceiling reduces length termination; it is useful only if the additional text
  improves quality-adjusted coverage rather than verbosity alone.
- H4: clustering conclusions will be threshold-sensitive; a condition is promotable only if its
  direction is stable across a meaningful threshold interval.

## Promotion rule

Promote one teacher condition only if it has no clear quality loss relative to the official setting,
improves semantic coverage on a majority of prompts, and does not rely on one uncalibrated cosine
threshold. If no condition passes, keep the official Qwen setting and report teacher-mode collapse
as a limitation rather than relabeling lexical variants as diverse modes.

## Threats to validity

- Internal: temperature and length are calibrated here; other decoding controls are fixed rather
  than exhaustively tuned.
- Construct: embedding clusters and an automatic 32B judge are proxies for expert assessments.
  Raw/instructed and threshold sensitivity are reported, but they do not replace human labels.
- Statistical: eight prompts provide a debugging/calibration signal, not a population estimate.
- External: prompts are length-stratified, not field-stratified, because the official TOMATO-Star
  release does not expose a domain label.
- Reproducibility: exact software/model/data revisions, prompt IDs, generation fingerprints, and
  output hashes are stored with the run.

## Verified source log

Checked 2026-08-02:

- **Qwen3 official quickstart** — software documentation. The Qwen team recommends, for
  non-thinking Qwen3, temperature 0.7, top-p 0.8, top-k 20, and min-p 0, and documents the hard
  `enable_thinking=False` switch. Relevance: center condition and request controls.
  <https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md>
- **SGLang official server-arguments documentation** — software documentation. It defines
  `--enable-deterministic-inference` as deterministic inference with batch-invariant operations.
  Relevance: reproducible sampling across dynamic batches.
  <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>
- **Qwen3-Embedding-4B official model card** — model documentation, Apache-2.0. It identifies text
  clustering as a supported task and recommends task-specific English instructions for downstream
  use. Relevance: predeclared raw-versus-instructed embedding sensitivity.
  <https://huggingface.co/Qwen/Qwen3-Embedding-4B>

