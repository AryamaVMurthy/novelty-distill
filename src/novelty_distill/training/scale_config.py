"""Render scale-equivalent training configs from the validated TOMATO-1k gate."""

import math
from collections.abc import Mapping
from typing import Any


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
