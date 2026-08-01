# Runtime environments

The official projects do not share one compatible dependency stack, so the project keeps them separate.

- `inference.in`: SGLang serving for teacher/evaluation/judge generation. SGLang 0.5.10 is pinned because Turing's observed RTX 6000 Ada node has driver 575.51.03 and CUDA 12.4; SGLang 0.5.13+ declares CUDA 13 dependencies.
- `training.in`: the exact package versions declared by the official OPSD repository, reused for TRL SFT/GKD and OPSD.
- GEM, MiniLLM, and DistiLLM run from their pinned official external checkouts. Any compatibility patch is stored as a small, reviewable patch without copying the upstream repository.

Compiled lock files are generated with `uv pip compile` after resolution is verified on Turing.
