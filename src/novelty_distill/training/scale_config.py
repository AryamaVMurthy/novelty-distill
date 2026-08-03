"""Render scale-equivalent training configs from the validated TOMATO-1k gate."""

import math
import re
from collections.abc import Mapping
from typing import Any

_TOMATO_MATRIX: tuple[tuple[str, str], ...] = (
    ("B1", "trl"),
    ("B2a", "trl"),
    ("B2b", "trl"),
    ("B2c", "trl"),
    ("B3", "trl"),
    ("B4", "gem"),
    ("C1-human", "trl"),
    ("C1-best1", "trl"),
    ("C1-diverse4", "trl"),
    ("C2-human", "trl"),
    ("C2-best1", "trl"),
    ("C2-diverse4", "trl"),
    ("C3", "distillm"),
    ("D1", "trl"),
    ("D2", "trl"),
    ("D3", "trl"),
    ("E2", "opsd"),
    ("E3", "opsd"),
    ("E4", "opsd"),
)


def build_promoted_training_manifest(
    *,
    model_profile: str,
    train_size: int,
    promoted_indices: tuple[int, ...],
    seeds: tuple[int, ...],
    training_dependencies: Mapping[tuple[int, int], str],
) -> dict[str, Any]:
    """Bind every promoted method/seed to its Slurm gate and deployable artifact."""

    valid_sizes = {
        "main": {5000, 20000},
        "replication": {1000, 5000, 20000},
    }
    if model_profile not in valid_sizes or train_size not in valid_sizes[model_profile]:
        raise ValueError("invalid model profile and training-size combination")
    if (
        not promoted_indices
        or len(promoted_indices) != len(set(promoted_indices))
        or any(index < 0 or index >= len(_TOMATO_MATRIX) for index in promoted_indices)
    ):
        raise ValueError("promoted indexes must be unique TOMATO matrix indexes")
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(seed not in {17, 29, 43} for seed in seeds)
    ):
        raise ValueError("training seeds must be unique members of 17, 29, and 43")
    expected_dependencies = {(index, seed) for seed in seeds for index in promoted_indices}
    if set(training_dependencies) != expected_dependencies:
        raise ValueError("training dependency mapping is not the full method-by-seed matrix")
    if any(
        re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", dependency) is None
        for dependency in training_dependencies.values()
    ):
        raise ValueError("training dependencies must be Slurm job or array-task IDs")

    if model_profile == "main":
        model = "Qwen/Qwen3-4B"
        revision = "1cfa9a7208912126459214e8b04321603b3df60c"
        run_infix = f"tomato{train_size}"
        target_name = f"teacher-targets-tomato{train_size}-v1.json"
    else:
        model = "Qwen/Qwen3-1.7B"
        revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        run_infix = f"qwen1p7b-tomato{train_size}"
        target_name = f"teacher-targets-qwen3-8b-tomato{train_size}-v1.json"

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for index in promoted_indices:
            baseline_id, backend = _TOMATO_MATRIX[index]
            run_name = f"{baseline_id}-{run_infix}-seed{seed}"
            checkpoint_root = f"checkpoints/{run_name}"
            model_path = model
            lora_path: str | None = f"{checkpoint_root}/final"
            model_dtype = "auto"
            generation_config = "configs/generation/eval_qwen3_4b_lora.yaml"
            if backend == "gem":
                model_path = checkpoint_root
                lora_path = None
                generation_config = "configs/generation/eval_qwen3_4b.yaml"
            elif backend == "distillm":
                steps = math.ceil(train_size / 4)
                checkpoint_root = f"{checkpoint_root}-l896-4gpu"
                model_path = f"{checkpoint_root}/{steps}"
                lora_path = None
                model_dtype = "bfloat16"
                generation_config = "configs/generation/eval_qwen3_4b.yaml"
            runs.append(
                {
                    "baseline_id": baseline_id,
                    "matrix_index": index,
                    "seed": seed,
                    "backend": backend,
                    "training_dependency": training_dependencies[(index, seed)],
                    "checkpoint_root": checkpoint_root,
                    "metadata_path": f"{checkpoint_root}/run_metadata.json",
                    "model_path": model_path,
                    "model_revision": revision,
                    "lora_path": lora_path,
                    "model_dtype": model_dtype,
                    "generation_config": generation_config,
                    "evaluation_id": f"{run_name}-temporal-k16",
                }
            )
    return {
        "schema_version": 1,
        "model_profile": model_profile,
        "train_size": train_size,
        "seeds": list(seeds),
        "promoted_indices": list(promoted_indices),
        "baseline_ids": [_TOMATO_MATRIX[index][0] for index in promoted_indices],
        "target_name": target_name,
        "runs": runs,
    }


def render_scale_training_config(
    base: Mapping[str, Any],
    *,
    backend: str,
    baseline_id: str,
    train_size: int,
    seed: int,
    model_profile: str = "main",
) -> dict[str, Any]:
    """Scale paths and exposure budgets without changing method hyperparameters."""

    if model_profile == "main" and train_size not in {5000, 20000}:
        raise ValueError("main scale training size must be 5000 or 20000")
    if model_profile == "replication" and train_size not in {1000, 5000, 20000}:
        raise ValueError("replication training size must be 1000, 5000, or 20000")
    if model_profile not in {"main", "replication"}:
        raise ValueError("model profile must be main or replication")
    if seed not in {17, 29, 43}:
        raise ValueError("scale training seed must be one of 17, 29, or 43")
    if backend not in {"trl", "opsd", "gem", "distillm"}:
        raise ValueError(f"unsupported scale backend {backend!r}")
    payload = dict(base)
    payload["baseline_id"] = baseline_id
    payload["input"] = f"data/tomato-open-train-{train_size}.jsonl"
    payload["max_examples"] = train_size
    payload["seed"] = seed
    if model_profile == "replication":
        run_name = f"{baseline_id}-qwen1p7b-tomato{train_size}-seed{seed}"
        target = f"data/teacher-targets-qwen3-8b-tomato{train_size}-v1.json"
        if backend == "distillm":
            payload["student_model"] = "Qwen/Qwen3-1.7B"
            payload["student_revision"] = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        else:
            payload["model"] = "Qwen/Qwen3-1.7B"
            payload["revision"] = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        if "teacher_model" in payload:
            payload["teacher_model"] = "Qwen/Qwen3-8B"
            payload["teacher_revision"] = "b968826d9c46dd6066d109eabc6255188de91218"
    else:
        run_name = f"{baseline_id}-tomato{train_size}-seed{seed}"
        target = f"data/teacher-targets-tomato{train_size}-v1.json"

    if backend == "distillm":
        if baseline_id != "C3":
            raise ValueError("DistiLLM scale config is only valid for C3")
        steps = math.ceil(train_size / int(payload["num_gpus"]))
        payload["teacher_targets"] = target
        payload["dev_examples"] = math.ceil(train_size * 0.04)
        payload["max_steps"] = steps
        payload["validation_interval"] = max(1, steps // 10)
        payload["raw_dir"] = f"data/distillm-{run_name}-l896/raw"
        payload["processed_dir"] = f"data/distillm-{run_name}-l896/processed"
        payload["output_dir"] = f"checkpoints/{run_name}-l896-4gpu"
        return payload

    steps = math.ceil(train_size / int(payload["gradient_accumulation_steps"]))
    payload["max_steps"] = steps
    if "save_steps" in payload:
        payload["save_steps"] = max(1, steps // 5)
    payload["output_dir"] = f"checkpoints/{run_name}"
    if "teacher_targets" in payload:
        payload["teacher_targets"] = target
    if backend == "gem":
        if baseline_id != "B4":
            raise ValueError("GEM scale config is only valid for B4")
        payload["tokenized_output"] = f"data/gem-{run_name}.jsonl"
    return payload
