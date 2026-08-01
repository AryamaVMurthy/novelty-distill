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

Nineteen executable training variants completed and wrote auditable `run_metadata.json` files:

- SFT/SeqKD/GEM: B1, B2a, B2b, B2c, B3, B4;
- off-policy distillation: all three C1 views, all three C2 views, and C3;
- on-policy GKD: D1, D2, D3;
- official OPSD controls: E2, E3, E4.

The runs use official TRL, GEM, DistiLLM, or OPSD code pinned in `third_party/manifest.yaml`.
E1 is intentionally fail-closed: the pinned official OPSD repository has no static-trajectory
off-policy mode, and this project does not invent an unofficial substitute.

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

## Next research-scale gate

The next run may scale only after fixed prompt IDs and decoding seeds are frozen. Use at least three
training seeds for promoted methods, then analyze prompt-level metric JSONL with
`scripts/analyze_paired_metrics.py` for paired bootstrap intervals, Cohen's dz, and Holm-adjusted
p-values. The smoke scores above must not be entered into a paper table.
