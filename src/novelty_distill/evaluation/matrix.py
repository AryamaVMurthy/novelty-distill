"""Strict alignment helpers for multi-model prompt-level evaluation artifacts."""

import math
from collections.abc import Mapping
from typing import Any


def collect_prompt_metric_rows(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float | str], ...]:
    """Flatten aligned evaluation artifacts into paired-analysis JSONL rows."""

    if len(evaluations) < 2:
        raise ValueError("evaluation collection requires at least two methods")
    methods = sorted(evaluations)
    if any(not method.strip() for method in methods):
        raise ValueError("method names must be non-empty")

    prompt_ids: set[str] | None = None
    metric_names: set[str] | None = None
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        payload = evaluations[method]
        if payload.get("schema_version") != 1:
            raise ValueError(f"method {method!r} has an unsupported evaluation schema")
        raw_prompts = payload.get("prompt_metrics")
        if not isinstance(raw_prompts, Mapping) or not raw_prompts:
            raise ValueError(f"method {method!r} has no prompt metrics")
        current_ids = {str(prompt_id) for prompt_id in raw_prompts}
        if "" in current_ids or len(current_ids) != len(raw_prompts):
            raise ValueError(f"method {method!r} has invalid prompt IDs")
        if prompt_ids is None:
            prompt_ids = current_ids
        elif current_ids != prompt_ids:
            raise ValueError(f"method {method!r} does not have exactly the reference prompt IDs")

        method_prompts: dict[str, dict[str, float]] = {}
        for prompt_id, raw_metrics in raw_prompts.items():
            if not isinstance(raw_metrics, Mapping) or not raw_metrics:
                raise ValueError(f"method {method!r} prompt {prompt_id!r} has no metrics")
            current_metrics = {str(name) for name in raw_metrics}
            if metric_names is None:
                metric_names = current_metrics
            elif current_metrics != metric_names:
                raise ValueError(
                    f"method {method!r} prompt {prompt_id!r} changed the metric names"
                )
            values: dict[str, float] = {}
            for name, raw_value in raw_metrics.items():
                if isinstance(raw_value, bool):
                    raise ValueError(f"metric {name!r} must be numeric, not boolean")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"metric {name!r} is not numeric") from error
                if not math.isfinite(value):
                    raise ValueError(f"metric {name!r} is not finite")
                values[str(name)] = value
            method_prompts[str(prompt_id)] = values
        normalized[method] = method_prompts

    assert prompt_ids is not None
    assert metric_names is not None
    return tuple(
        {
            "method": method,
            "prompt_id": prompt_id,
            **{
                metric: normalized[method][prompt_id][metric]
                for metric in sorted(metric_names)
            },
        }
        for method in methods
        for prompt_id in sorted(prompt_ids)
    )
