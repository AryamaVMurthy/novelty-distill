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
) -> dict[str, Any]:
    """Scale paths and exposure budgets without changing method hyperparameters."""

    if train_size not in {5000, 20000}:
        raise ValueError("scale training size must be 5000 or 20000")
    if seed not in {17, 29, 43}:
        raise ValueError("scale training seed must be one of 17, 29, or 43")
    if backend not in {"trl", "opsd", "gem", "distillm"}:
        raise ValueError(f"unsupported scale backend {backend!r}")
    payload = dict(base)
    payload["baseline_id"] = baseline_id
    payload["input"] = f"data/tomato-open-train-{train_size}.jsonl"
    payload["max_examples"] = train_size
    payload["seed"] = seed
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
