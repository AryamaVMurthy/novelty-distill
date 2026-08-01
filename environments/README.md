# Runtime environments

The official projects do not share one compatible dependency stack, so the project keeps them separate.

- `data.in`: the official Hugging Face `datasets` loader plus this adapter's two runtime dependencies.
- `inference.in`: the minimal SGLang server for teacher/evaluation/judge generation. SGLang 0.5.10 is pinned because Turing's observed RTX 6000 Ada node has driver 575.51.03 and CUDA 12.4; SGLang 0.5.13+ declares CUDA 13 dependencies.
- `trl.in`: the minimal official TRL SFT/GKD stack.
- `training.in`: the exact package versions declared by the official OPSD repository.
- `gem.in`: the minimal imports used by the pinned official GEM training entrypoint.
- `distillm.in`: the minimal imports used by the pinned official DistiLLM preprocessor and trainer.
- MiniLLM runs from its pinned official external checkout. Any compatibility patch is stored as a small, reviewable patch without copying the upstream repository.

Compiled lock files are generated with `uv pip compile` after resolution is verified on Turing.
