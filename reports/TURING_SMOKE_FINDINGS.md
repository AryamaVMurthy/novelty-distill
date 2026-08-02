# Turing smoke findings

This report records implementation evidence, not headline research results. Training runs use one
or two records and one optimizer step; official evaluation runs use one or two benchmark samples.

## Verified setup

- Turing node01: NVIDIA RTX 6000 Ada Generation, 49,140 MiB VRAM, CUDA 12.4 module.
- Canonical TOMATO-Star open-generation subset: 1,000 records, dataset revision
  `fcd201d92758a642465a7653b8055f3a04d5f439`, SHA-256
  `995c9648a9b591fbae381a2f3f2de265f25a85ebb3e7ba891309403692ae9ccf`.
- Teacher path: official Qwen3-14B through SGLang, eight samples per prompt; fixed samples were
  scored with official Qwen3-32B-FP8 and clustered with official Qwen3-Embedding-4B.
- Storage, environments, caches, logs, data, generations, and checkpoints live under
  `/scratch/aryama.murthy/novelty-distill`.

## Baseline execution

Nineteen executable training variants reached real optimizer steps and wrote auditable
`run_metadata.json` files:

- SFT/SeqKD/GEM: B1, B2a, B2b, B2c, B3, B4;
- off-policy distillation: all three C1 views, all three C2 views, and C3;
- on-policy GKD: D1, D2, D3;
- official OPSD controls: E2, E3, E4.

The runs use official TRL, GEM, DistiLLM, or OPSD code pinned in `third_party/manifest.yaml`.
E1 is intentionally fail-closed: the pinned official OPSD repository has no static-trajectory
off-policy mode, and this project does not invent an unofficial substitute. The registry records
that boundary with `execution_status: fail_closed`, so matrix consumers can distinguish an
unsupported planned control from a missing or silently skipped run.

A post-smoke source audit reopened the E2/E3 task-faithfulness gate: OPSD's upstream data collator
wraps every input as a mathematics problem and requests step-by-step reasoning with a boxed answer.
That smoke established trainer/loss deployability, but it is not valid TOMATO treatment evidence.
The corrected adapter keeps the official trainer and loss, renders the E2/E3/E4 student prompt
identically, and adds historical hypotheses/inspirations only to the privileged teacher prompt.
A fresh main-model smoke must validate this corrected collator before E2/E3 production artifacts
are accepted.

Teacher-generation jobs 17918 and 17922 independently produced the same eight-record projection
for one fixed TOMATO prompt. All eight hypotheses were distinct within each run, and the canonical
text-list SHA-256 was
`f1e2a65adbee57edb7c3a466e8366d725c850e1f561d1d7df9657b5451901be3`. This required one
single-sample SGLang request per hypothesis with seeds `base_seed + sample_index`; one deterministic
request with `n=8` had incorrectly returned eight copies of one hypothesis. Judge job 17926 then
completed against the corrected shard using Qwen3-32B-FP8, producing eight auditable request IDs
and a quality score of 0.95 for every response. Cluster job 17931 assigned all eight responses to
one semantic mode at the frozen cosine threshold of 0.82, while still producing the random-1,
best-1, common-mode-1, and diverse-4 target views in the isolated
`data/teacher-targets-17918.json` artifact.

All eight corrected teacher responses reached the configured 512-token ceiling. They are valid
training records, but this saturation must be measured on a larger prompt sample before promotion;
the final length cap or prompt concision constraint should be frozen from that analysis.

## Official evaluation evidence

NoveltyBench jobs 17906 and 17908 used pinned Inspect Evals commit
`6a35510e530f236fd1dbcd9df888f01937c8494a` and Qwen3-4B served by SGLang. Across the same two
prompts and two generations per prompt, all four completions were non-empty, none began with
`<think>`, and the official smoke scores were `distinct_k=1.0` and `utility_k=5.0`. Prompt IDs,
completion bytes, per-prompt scores, and aggregate scores were identical between runs. The
canonical completion SHA-256 is
`b0bd715292216604d7c07f179216ea12818abfea29e8f28f34804ed8fe2c2de1`. The artifacts are:

- `evaluations/noveltybench/2026-08-01T23-41-54-00-00_novelty-bench_hWqi5QnUNLXP2dwEg9SPe3.eval`
- `evaluations/noveltybench/2026-08-01T23-43-04-00-00_novelty-bench_cYWPyunqzfTH7HMXtHcLV9.eval`

Pinned official HypoSpace smoke results with Qwen3-4B were:

| Domain | Parse | Validity | Novelty | Recovery |
|---|---:|---:|---:|---:|
| causal | 100% | 0% | 100% | 0% |
| 3D | 100% | 0% | 100% | 0% |
| Boolean | 100% | 100% | 50% | 50% |

These values prove the provider, parser, validators, and scorers execute. Their sample sizes are too
small for model comparisons.

## Findings that changed the implementation

1. Qwen thinking mode invalidated both evaluations by consuming the output budget with reasoning.
   Request-level non-thinking mode fixed NoveltyBench output usage from 512-token `<think>` traces
   to valid answer text and raised HypoSpace causal parsing from 0% to 100%.
2. HypoSpace's upstream client can absorb provider exceptions. The thin adapter now validates the
   official result object and fails when any provider error was recorded.
3. Official packages require isolated pinned environments. In particular, the NoveltyBench lock's
   newest Torch/CUDA resolution was incompatible with Turing's driver, so the known-good official
   PyTorch 2.9.1 CUDA 12.8 wheel is pinned as a supplemental runtime dependency.
4. Turing scratch is node-local and home is full. Submissions and logs must be routed through
   node01 scratch; `scripts/turing_submit.sh` makes that path reproducible.
5. Inspect's seed does not freeze a task's pre-evaluation random shuffle, and request seeds alone do
   not make dynamic batches invariant. NoveltyBench now uses the official `shuffle=false` task
   option, and every SGLang server uses `--enable-deterministic-inference`. SGLang ports are derived
   from the Slurm job ID so concurrent jobs cannot connect to one another's server.
6. SGLang deterministic mode plus one request containing `n=8` makes every choice identical for
   this Qwen path. Independent single-sample requests with derived seeds preserve exact rerun
   reproducibility while recovering eight distinct hypotheses. The strategy is part of the shard
   fingerprint so old collapsed shards fail validation instead of being resumed.
7. Lexically distinct samples are not necessarily distinct scientific modes. Both prompts in the
   original two-prompt teacher smoke had eight unique texts but only one embedding cluster each;
   the corrected deterministic prompt showed the same result. The 32B judge was also nearly
   saturated (0.90--1.00 across the original prompts, exactly 0.95 on the corrected prompt).
   Teacher-temperature, clustering-threshold, and judge-calibration checks are therefore required
   before calling `diverse4` a semantic-diversity treatment at research scale.

## Next research-scale gate

The next run may scale only after fixed prompt IDs and decoding seeds are frozen. Use at least three
training seeds for promoted methods, then analyze prompt-level metric JSONL with
`scripts/analyze_paired_metrics.py` for paired bootstrap intervals, Cohen's dz, and Holm-adjusted
p-values. First calibrate the teacher-mode threshold and verify that the quality judge has useful
within-prompt resolution on a larger stratified prompt sample. The smoke scores above must not be
entered into a paper table.
