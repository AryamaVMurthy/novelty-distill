# Same-prompt embedding geometry protocol

## Status and motivation

This protocol was frozen on 2026-08-03 before any TOMATO temporal-test metric for a trained
student was available. It is a **secondary descriptive analysis** and cannot change the primary
contrast family or promotion decision by itself.

Chen, Zhao, and Cohan, *Measuring the Gap Between Human and LLM Research Ideas* (arXiv:2607.01233),
compare prompt-matched human and model ideas in embedding space as one mechanism analysis for the
research-taste gap. We adapt that question to distillation: does a trained student's output cloud
remain geometrically aligned with the external teacher, move toward the historical human idea, or
merely become more concentrated? The paper's task and embedding pipeline differ from ours, so its
reported geometry is motivation rather than an expected value.

## Frozen quantities

For temporal prompt `p`, let `S_p` be a method's 16 embeddings, `T_p` the A1 teacher's 16
embeddings, and `H_p` the single A3 historical-human embedding. All vectors use the existing pinned
Qwen3-Embedding-4B model, revision, instruction, truncation limit, pooling rule, and normalized
float32 cache used by the primary semantic-mode evaluator.

For each method and prompt, compute:

- teacher affinity: the mean cosine over every pair in `S_p × T_p`;
- human affinity: the mean cosine over every pair in `S_p × H_p`;
- teacher-minus-human affinity: teacher affinity minus human affinity;
- within-method concentration: the mean cosine over all distinct unordered pairs in `S_p`.

Positive teacher-minus-human affinity means closer to A1 than A3 in this representation. Higher
within-method cosine means a tighter output cloud; it is not a quality or scientific-validity
score. Each trained method also reports paired prompt-level deltas from untouched A0.

## Estimation and execution

Point estimates average prompt metrics, giving every temporal research problem equal weight.
Uncertainty uses 10,000 prompt-level bootstrap resamples at seed 17 with pointwise 95% percentile
intervals. Samples within a prompt are never treated as independent research problems.

The analyzer is CPU-only and reads embeddings already written by the frozen evaluator. It fails if
any requested cache vector, score shard, prompt, sample, model revision, or cache manifest is
missing or incompatible; it never launches replacement embedding inference. Inputs and cache
manifests are content-hashed in the result. The analysis runs for the primary 1k matrix, matched
1x/4x exposure sensitivity, DRKL treatment, and promoted multi-seed stages.

## Interpretation boundaries

- Embedding proximity is operational semantic similarity, not expert-confirmed novelty.
- A method can become more human-aligned by cosine while losing quality, feasibility, semantic
  modes, or research-taste breadth. All outcomes must be inspected together.
- A3 is one realized historical hypothesis, whereas A1 and students have 16 stochastic samples.
  Cross-set affinity averages all available pairs, while within-method concentration is defined
  only for multi-sample candidates.
- The embedding instruction emphasizes mechanism, intervention, and experimental test; results do
  not establish lexical, topical, or causal similarity under other representations.
- The analysis is post-freeze secondary evidence and adds no confirmatory p-value family.
