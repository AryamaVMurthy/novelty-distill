# Forward Semantic Seed Distillation: short-run design

## Objective

Find a small, reproducible modification to the existing Qwen3-14B to Qwen3-4B
distillation pipeline that improves quality-qualified semantic breadth without
reducing feasibility or soundness. The method should test the two mechanisms
suggested by *Roll the Dice & Look Before You Leap*: move randomness to the
input and expose the model to information about the future trajectory before
ordinary next-token generation.

The experiment is a post-freeze exploratory search. Existing corrected
three-seed results remain the baseline evidence and are not mixed into a new
preregistered claim.

## Design principles

1. Keep off-policy forward KL (`C1-best1`) as the token-level teaching loss.
   It is the strongest current baseline and isolates the new mechanism from KL
   direction and on-policy sampling.
2. Treat the frozen semantic embedding as a geometric instrument, not a
   differentiable text generator or a ground-truth novelty oracle.
3. Put stochasticity before decoding. A seed selects a trajectory once; normal
   low-temperature decoding then realizes it coherently.
4. Give probability only to teacher trajectories that pass a validity gate.
   Semantic rarity cannot compensate for an unsound hypothesis.
5. Compare methods at matched optimizer-example exposure and generation budget.

## Candidate methods

### A. Coverage-residual forward KL (CR-FKL)

For prompt `x`, let the eight frozen teacher responses have normalized
embeddings `z_T[i]` and validity weights `q[i]`. Generate a small fixed probe
set from the untouched student and embed it as `z_S[j]`. Estimate student
coverage at each teacher response with a cosine/von-Mises--Fisher kernel:

```
rho_S(i) = mean_j exp(kappa * (cos(z_T[i], z_S[j]) - 1))
```

Choose the training trajectory using:

```
w[i] proportional to validity(q[i]) * (rho_S(i) + epsilon)^(-gamma)
```

Then apply the existing off-policy forward-KL loss on the selected teacher
trajectory. This tests whether teaching valid, student-missing futures improves
coverage without changing the distillation loss.

### B. Gaussian seed conditioning (GSC-FKL)

Draw a prompt-level vector `u ~ N(0, I_r)`, quantize each coordinate into a
small balanced alphabet, and serialize the result in a short input prefix. The
seed is inserted after the fixed task instruction and before the research
question. It is sampled once per response and held fixed for the entire
generation.

Training exposes the same prompt to multiple valid teacher responses with
different seeds. Inference varies the seed while keeping output sampling colder
than the existing `temperature=0.7` protocol. This isolates early trajectory
randomness from ordinary token-level temperature noise.

The first implementation uses text tokens rather than modifying the embedding
layer. That keeps SGLang evaluation compatible and makes the intervention
auditable. A continuous soft-prefix projector is deferred unless the text-seed
pilot shows a seed-dependent effect.

### C. Forward-plan seed conditioning (FPS-FKL)

Fit PCA only on training teacher embeddings. Represent each teacher response by
a low-dimensional future code consisting of quantized PCA coordinates plus a
small Gaussian perturbation. Serialize that code in the same prefix format as
GSC-FKL. The code is not described as a topic or answer; it is a compact future
trajectory coordinate.

At inference, sample from the fitted mixture of valid teacher codes rather than
from an isotropic Gaussian. This combines:

- early input randomness (roll first),
- a coarse representation of the future semantic destination (look first), and
- ordinary forward-KL token teaching (realize the selected trajectory).

FPS-FKL is the preferred candidate if it outperforms isotropic seed
conditioning. CR-FKL can be combined with FPS-FKL by oversampling valid future
codes that the untouched student undercovers.

### D. Hybrid teacherless lookahead distillation (TL-FKL)

Add a low-weight teacherless auxiliary objective to ordinary forward KL. For
the auxiliary pass, replace the teacher-response prefix with a repeated neutral
dummy token while retaining the teacher response as the label sequence. Each
response position must therefore be predicted from the prompt, seed, position,
and global hidden computation rather than from the preceding ground-truth
response tokens:

```
L = L_forward_KL(ordinary teacher-forced trajectory)
    + lambda_TL * L_CE(prompt + dummy response positions, teacher response)
```

This follows the actual multi-token/teacherless mechanism in *Roll the Dice &
Look Before You Leap* more closely than a shifted-next-token loss. The
teacherless pass is computed sequentially after the forward-KL pass to avoid
doubling peak activation memory. It is an auxiliary representation-learning
signal only; inference remains standard autoregression.

The pilot tests `lambda_TL` in a very small declared set. If optimization is
non-finite, context construction is ambiguous, or quality falls through the
safety gate, this branch is rejected rather than replaced with an unprincipled
future-token surrogate.

## Successive-halving experiment

### Gate 0: artifact and geometry audit

- Validate the existing 1,000-prompt teacher bank, scores, clusters, and
  embeddings on Turing.
- Reuse existing untouched-student generations if their decoding contract
  matches; otherwise generate a bounded probe set.
- Fit PCA on training artifacts only.
- Check that quantized codes are balanced, reproducible, and not predictable
  from response quality alone.

### Gate 1: decoding-only pilot

On 64 fixed held-out prompts, generate `K=8` responses from the untouched
student under:

1. ordinary sampling at temperature 0.7;
2. colder sampling with isotropic Gaussian seed codes;
3. colder sampling with future-mixture seed codes.

This does not test learned seed use, but rejects prefix formats that damage
instruction following or merely reproduce temperature sampling. Promote a
seed format only if feasibility and soundness stay within 0.10 points of the
ordinary control and within-prompt semantic spread increases.

### Gate 2: short training pilot

Use a fixed 128-prompt training subset and at most 32 optimizer steps. Compare:

1. `C1-best1-short`;
2. `CR-FKL-short`;
3. `GSC-FKL-short`;
4. `FPS-FKL-short`;
5. `TL-FKL-short`;
6. `GSC-TL-FKL-short` only if both components independently pass;
7. `CR-FPS-TL-FKL-short` only if each component independently passes.

Evaluate on a disjoint 128-prompt set with `K=4`. Use the frozen Qwen judge and
embedding pipeline. Rank methods first by feasibility/soundness safety, then by
quality-qualified semantic yield. Raw diversity cannot promote a method.

### Gate 3: confirmation

Train only the best method and `C1-best1` on TOMATO-1k with the existing
1,000-example exposure budget and seeds 17, 29, and 43. Evaluate with the
corrected TOMATO protocol. Run the corrected NoveltyBench and HypoSpace suites
only after the TOMATO gate succeeds.

No full 5k baseline matrix is required. If 1k confirmation succeeds, scale only
the proposed method and `C1-best1` to 5k.

## Selection rule

A candidate is eligible only when:

- generation, scoring, and embedding artifacts validate completely;
- feasibility and soundness do not fall by more than 0.10 versus the matched
  forward-KL control in the short pilot;
- quality-qualified semantic yield improves;
- improvement persists over the declared cosine/kernel sensitivity range;
- seed-conditioned outputs demonstrably depend on the seed, measured by
  within-prompt between-seed versus repeated-same-seed distance.

If no candidate clears the safety gate, the result is that early noise does not
transfer safely in this setup; the workflow must not select the least-bad run
and label it successful.

## Provenance and failure handling

Every Turing job records the Git commit, full rendered config, Slurm job ID,
node and GPU, package versions, random seeds, prompt IDs, teacher-target hash,
checkpoint hash, and generation/scoring manifests. Jobs fail closed on missing
teacher artifacts, incomplete prompts, non-finite losses, seed-code mismatch,
or incompatible decoding contracts.

Large artifacts remain under node-local scratch. Only compact manifests,
summaries, and final adapters are candidates for persistent storage.

## Main claim boundary

The strongest permissible claim is improvement in quality-qualified semantic
coverage under the frozen operational protocol. Neither Gaussian seed distance,
embedding novelty, NoveltyBench, nor an LLM judge alone establishes scientific
novelty. Human semantic-boundary calibration and literature-grounded expert
review remain necessary for a scientific-creativity claim.
