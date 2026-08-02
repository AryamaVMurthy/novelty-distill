# Novelty Distillation on Turing — Implementation Plan

## Goal

Implement the complete baseline comparison for scientific-idea novelty:

- supervised fine-tuning;
- sequence-level/off-policy distillation;
- on-policy distillation;
- privileged on-policy self-distillation;
- quality and idea-mode coverage evaluation.

Use official models, datasets, trainers, and author repositories wherever available. Write custom code only for TOMATO-Star formatting, privileged-context wiring, experiment configuration, and project-specific novelty metrics.

## Official components only

| Need | Official source | Use |
|---|---|---|
| Models | [Qwen on Hugging Face](https://huggingface.co/Qwen) | Qwen3 1.7B/4B/8B/14B/32B and Qwen3-Embedding-4B |
| Inference | [SGLang](https://github.com/sgl-project/sglang) | teacher generation, evaluation sampling, and judge serving |
| SFT | [Hugging Face TRL](https://github.com/huggingface/trl) | `SFTTrainer` |
| SeqKD and GKD | [TRL GKDTrainer](https://huggingface.co/docs/trl/gkd_trainer) | official premade off-policy/on-policy trainer |
| OPSD | [siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD) | privileged self-distillation reference and trainer |
| MiniLLM | [microsoft/LMOps/minillm](https://github.com/microsoft/LMOps/tree/main/minillm) | reverse-KL reference |
| DistiLLM | [jongwooko/distillm](https://github.com/jongwooko/distillm) | skew-KL/adaptive off-policy baseline |
| GEM | [liziniu/GEM](https://github.com/liziniu/GEM) | diversity-preserving SFT baseline |
| UltraFeedback | [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | pipeline pilot |
| TOMATO-Star | [ZonglinY/TOMATO-Star](https://huggingface.co/datasets/ZonglinY/TOMATO-Star) | main training and temporal evaluation data |
| NoveltyBench | [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals) | official premade NoveltyBench evaluation |
| HypoSpace | [CTT-Pavilion/_HypoSpace](https://github.com/CTT-Pavilion/_HypoSpace) | exact validity/uniqueness/recovery evaluation |
| Tracking | [Weights & Biases SDK](https://github.com/wandb/wandb) | configs, runs, tables, and artifacts |

Every dependency or external repository must be pinned to a commit/release in `third_party/manifest.yaml`. Do not copy unofficial reimplementations or code from blogs.

## Models

| Role | Model |
|---|---|
| Student | `Qwen/Qwen3-4B` |
| External teacher | `Qwen/Qwen3-14B` |
| Self-teacher | frozen `Qwen/Qwen3-4B` |
| Judge | `Qwen/Qwen3-32B` |
| Embeddings | `Qwen/Qwen3-Embedding-4B` |
| Small replication | student 1.7B, teacher 8B |

Use instruction-tuned Qwen3 with `enable_thinking=False`, a fixed chat template, fixed output length, and pinned model/tokenizer revisions.

## Complete baseline matrix

### Controls

| ID | Baseline |
|---|---|
| A0 | untouched Qwen3-4B |
| A1 | Qwen3-14B teacher |
| A2 | optional Qwen3-4B-Base |
| A3 | historical TOMATO-Star hypothesis |

### SFT and hard sequence distillation

| ID | Baseline | Official implementation |
|---|---|---|
| B1 | human-target SFT | TRL `SFTTrainer` |
| B2a | teacher random-1 SeqKD | TRL `SFTTrainer` or `GKDTrainer(seq_kd=True)` |
| B2b | teacher best-of-8 SeqKD | same |
| B2c | teacher most-common-mode SeqKD | same |
| B3 | teacher diverse-4 SFT | TRL `SFTTrainer` |
| B4 | GEM | official `liziniu/GEM` |

### Off-policy soft distillation

| ID | Baseline | Official implementation |
|---|---|---|
| C1 | off-policy forward KL | TRL `GKDTrainer`, static trajectories, `lmbda=0`, forward direction |
| C2 | off-policy reverse KL | TRL/MiniLLM official implementation, static trajectories |
| C3 | DistiLLM | official `jongwooko/distillm` |

Run C1/C2 on human, teacher-best-1, and teacher-diverse trajectories through configuration, not separate trainer code.

### External-teacher on-policy distillation

| ID | Baseline | Official implementation |
|---|---|---|
| D1 | GKD forward KL | TRL `GKDTrainer`, student trajectories, `lmbda=1` |
| D2 | GKD reverse KL | TRL `GKDTrainer`, reverse direction |
| D3 | GKD generalized JSD | TRL `GKDTrainer`, configured `beta` |

### Self-distillation

| ID | Baseline | Official implementation |
|---|---|---|
| E1 | privileged off-policy self-KD | fail closed: pinned official OPSD has no static-trajectory mode; no unofficial substitute |
| E2 | privileged OPSD forward KL | official `siyan-zhao/OPSD`, minimally adapted to TOMATO context |
| E3 | privileged OPSD reverse KL | official OPSD loss configuration/adaptation |
| E4 | OPSD without privileged context | official OPSD trainer control |

The main custom adaptation is a TOMATO batch builder that gives the student the ordinary prompt and gives the frozen self-teacher the same prompt plus the historical hypothesis/inspirations.
E1 remains in the registry as a planned negative-capability control, with a machine-readable
`fail_closed` status and reason. It is excluded from executable matrices unless a reviewed official
upstream implementation becomes available.

## Repository scaffold

```text
novelty-distill/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── uv.lock
├── third_party/
│   └── manifest.yaml              # official URLs, commits, licenses
├── configs/
│   ├── models/
│   ├── data/
│   ├── methods/                   # one config per A/B/C/D/E baseline
│   ├── eval/
│   └── experiments/
├── src/novelty_distill/
│   ├── data/
│   │   ├── schema.py
│   │   ├── ultrafeedback.py
│   │   ├── tomato.py
│   │   └── teacher_views.py
│   ├── integrations/
│   │   ├── sglang.py
│   │   ├── trl.py
│   │   ├── opsd.py
│   │   ├── gem.py
│   │   ├── minillm.py
│   │   └── distillm.py
│   ├── generation/
│   ├── evaluation/
│   │   ├── semantic_modes.py
│   │   ├── quality.py
│   │   └── statistics.py
│   └── tracking/
├── scripts/
│   ├── bootstrap_official.sh
│   ├── prepare_data.py
│   ├── generate_teacher.py
│   ├── train.py
│   └── evaluate.py
├── slurm/
│   ├── probe.sbatch
│   ├── prepare_data.sbatch
│   ├── generate_teacher.sbatch
│   ├── train.sbatch
│   └── evaluate.sbatch
└── tests/
    ├── test_data.py
    ├── test_official_integrations.py
    ├── test_masks.py
    ├── test_teacher_frozen.py
    └── test_resume.py
```

Integration files must remain thin wrappers around official APIs. Do not rewrite official trainers inside this repository.

## Data preparation

### UltraFeedback pilot

Create the official four views:

- random-1;
- best-1;
- all-4;
- diverse-2.

Use this only to verify SFT, SeqKD, GKD, resume, logging, and evaluation.

### TOMATO-Star main data

Use the official raw `ZonglinY/TOMATO-Star` dataset, not its separate generated SFT dataset.

Create:

1. open generation: `research_question + background_survey → fine_grained_hypothesis`;
2. composition: `research_question + background_survey + inspiration → fine_grained_hypothesis`.

Remove title, DOI, source ID, authors, venue, and inspiration-paper title from student prompts. Keep the official temporal test split untouched.

Use fixed subsets of 1k, 5k, and 20k training IDs. All baselines use exactly the same IDs.

## Teacher generation

Use Qwen3-14B to generate eight responses for every selected training prompt. Save them once, then derive index views:

- random-1;
- best-1;
- common-mode-1;
- diverse-4;
- all-8.

Use the official Qwen tokenizer/chat template and the official Qwen embedding model. Never regenerate separately for different baselines.

## Basic evaluation

For each final model, generate K=16 responses using the same prompts, temperatures, top-p, maximum tokens, and seeds.

Report:

- quality and feasibility from the fixed Qwen3-32B judge;
- semantic cluster count;
- teacher ModeRecall@16 and ModePrecision@16;
- ClusterJSD;
- quality-adjusted coverage;
- nearest-training-target similarity;
- NoveltyBench `distinct_k` and `utility_k` from the official Inspect Evals implementation;
- HypoSpace Validity, Uniqueness, and Recovery from the official repository.

Use three training seeds for the main baselines. Run reverse-KL/JSD and optional controls with one seed first, then promote them after the implementation is stable.

## Turing execution order

1. **Route storage:** keep all project artifacts and logs in node01-local `/scratch/$USER`; submit with `scripts/turing_submit.sh`. Do not place experiment outputs in the full home filesystem.
2. **Bootstrap:** clone this repo; install pinned packages; fetch only pinned official repositories.
3. **Probe:** submit a short Slurm job to record GPU model, VRAM, CUDA, and which model combinations fit.
4. **Pilot:** run untouched evaluation and 1k UltraFeedback SFT/SeqKD/GKD.
5. **Prepare TOMATO:** create the canonical 1k/5k/20k datasets and leakage checks.
6. **Generate teacher data:** produce eight Qwen3-14B outputs per prompt once.
7. **Run baselines:** 1k smoke test, 5k development, then 20k final training.
8. **Evaluate:** TOMATO temporal test, NoveltyBench, and HypoSpace.
9. **Replicate:** run the central baselines with Qwen3-1.7B/8B after the 4B/14B study works.

Use `sbatch` for all model, data, and evaluation work. The login node is only for Git, syncing, submission, queue checks, and reading logs. Store code/configs/small adapters in home and models/datasets/generations/checkpoints in `/scratch/$USER`.

## Implementation rules

- Pin every official repository and dependency before experiments.
- Preserve official license and attribution files.
- Prefer configuration over copied or duplicated trainer code.
- Use the same student initialization and training-token budget for comparable baselines.
- Freeze teacher parameters and test that they never receive gradients.
- Test loss masks and student-generated prefixes before any full run.
- Log Git commit, config, model/data revisions, seed, Slurm job ID, GPU, versions, and artifact paths.
- Fail explicitly on missing data, CUDA errors, OOM, corrupt checkpoints, or unsupported official APIs.
- Do not silently quantize, change model size, replace a baseline, or use an unofficial implementation.

## Completion criteria

The implementation is ready when:

- every baseline above runs through one config-driven command;
- official integrations are pinned and tested;
- all baselines share the same data and evaluation formats;
- training can resume correctly;
- teacher data is generated only once and reused;
- final outputs reproduce from saved configs, commits, seeds, and artifacts.
