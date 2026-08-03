# OPSD loss-validity audit

## Scope

This audit covers the optional `jsd_token_clip` path in the pinned external OPSD trainer at commit
`7448751f307a9cdbcc1246dd1565a1a605b443df`. It does not change the author's generalized-JSD/KL
implementation or claim that the unclipped objective is a new method.

## Failure mechanism

For the forward-KL endpoint, the trainer first computes one signed contribution per vocabulary
item,

\[
  d_i = q_i(\log q_i - \log p_i),
\]

and only the vocabulary sum \(\sum_i d_i = D_{KL}(q\|p)\) is guaranteed to be nonnegative. The
optional path applies `clamp(max=jsd_token_clip)` to each \(d_i\) before the vocabulary sum. It
therefore caps positive terms while retaining negative terms and no longer computes a divergence.
For example, with \(q=(0.5,0.5)\), \(p=(0.9,0.1)\), and a 0.05 cap, the valid contributions are
approximately \((-0.2939, 0.8047)\), whose sum is \(0.5108\). Elementwise upper clipping instead
produces approximately \((-0.2939, 0.05)\), whose sum is \(-0.2439\). The reverse-KL endpoint and
interior generalized-JSD mixtures have the same signed-contribution issue.

## Observed evidence

Production E2 job 18561 used `jsd_token_clip: 0.05`. Its first 18 optimizer steps were finite and
had nonzero gradients, but every logged objective was negative; the observed values were roughly
-0.0107 to -0.0157. This is direct evidence of the invalid clipped estimator, not evidence that the
student surpassed the teacher. The job was cancelled before checkpoint 25. Its three saved
trajectory logs were moved intact to:

`checkpoints-invalid-vocab-clip-v1/E2-tomato1k-seed17-job18561`

No invalid adapter or optimizer checkpoint is reused.

## Repair and gate

Commit `c0f9615` makes `jsd_token_clip` optional and sets it to `null` in every OPSD run
configuration. This exercises the pinned trainer's existing unclipped generalized-JSD/KL path.
The change was developed with a failing configuration-contract test and the full local suite then
passed with 266 tests and two expected PyTorch-dependent skips.

Qwen3-4B gate 18597 completed one real E2 optimizer step at the same pinned model and OPSD commit.
It recorded:

- loss and on-policy loss: 1.215773;
- gradient norm: 2.714494;
- runtime: 23.542 seconds;
- `jsd_token_clip: null` in saved run metadata;
- a loadable final LoRA adapter.

The gate establishes a finite, nonnegative, differentiable execution path. It does not establish
downstream idea quality. E2/E3/E4 must be trained afresh under this contract and evaluated through
the same frozen temporal protocol as every other baseline.

## Interpretation boundary

The upstream comment motivates the optional cap as protection against individual style-token
contributions, but its current placement is before the vocabulary reduction. A future robust-loss
sensitivity would need a separately reviewed definition, such as clipping the already-summed
nonnegative divergence per sequence position. It must not be silently folded into the primary
OPSD baseline.
