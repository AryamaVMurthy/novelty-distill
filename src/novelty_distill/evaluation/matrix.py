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


def collect_threshold_metric_rows(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float | str], ...]:
    """Flatten complete, aligned threshold curves into prompt-level JSONL rows."""

    if len(evaluations) < 2:
        raise ValueError("threshold collection requires at least two methods")
    methods = sorted(evaluations)
    if any(not method.strip() for method in methods):
        raise ValueError("method names must be non-empty")

    expected_thresholds: set[str] | None = None
    expected_prompt_ids: set[str] | None = None
    expected_metric_names: set[str] | None = None
    normalized: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for method in methods:
        payload = evaluations[method]
        if payload.get("schema_version") != 1:
            raise ValueError(f"method {method!r} has an unsupported evaluation schema")
        raw_curve = payload.get("prompt_metrics_by_threshold")
        if not isinstance(raw_curve, Mapping) or not raw_curve:
            raise ValueError(f"method {method!r} has no threshold prompt metrics")
        thresholds = {str(threshold) for threshold in raw_curve}
        if "" in thresholds or len(thresholds) != len(raw_curve):
            raise ValueError(f"method {method!r} has invalid thresholds")
        try:
            numeric_thresholds = tuple(float(threshold) for threshold in thresholds)
        except ValueError as error:
            raise ValueError(f"method {method!r} has non-numeric thresholds") from error
        if any(not math.isfinite(value) for value in numeric_thresholds):
            raise ValueError(f"method {method!r} has non-finite thresholds")
        if expected_thresholds is None:
            expected_thresholds = thresholds
        elif thresholds != expected_thresholds:
            raise ValueError(f"method {method!r} does not have exactly the reference thresholds")

        method_curve: dict[str, dict[str, dict[str, float]]] = {}
        for raw_threshold, raw_prompts in raw_curve.items():
            threshold = str(raw_threshold)
            if not isinstance(raw_prompts, Mapping) or not raw_prompts:
                raise ValueError(f"method {method!r} threshold {threshold!r} has no prompts")
            prompt_ids = {str(prompt_id) for prompt_id in raw_prompts}
            if "" in prompt_ids or len(prompt_ids) != len(raw_prompts):
                raise ValueError(
                    f"method {method!r} threshold {threshold!r} has invalid prompt IDs"
                )
            if expected_prompt_ids is None:
                expected_prompt_ids = prompt_ids
            elif prompt_ids != expected_prompt_ids:
                raise ValueError(
                    f"method {method!r} threshold {threshold!r} does not have exactly "
                    "the reference prompt IDs"
                )

            threshold_prompts: dict[str, dict[str, float]] = {}
            for raw_prompt_id, raw_metrics in raw_prompts.items():
                prompt_id = str(raw_prompt_id)
                if not isinstance(raw_metrics, Mapping) or not raw_metrics:
                    raise ValueError(
                        f"method {method!r} threshold {threshold!r} prompt "
                        f"{prompt_id!r} has no metrics"
                    )
                metric_names = {str(name) for name in raw_metrics}
                if expected_metric_names is None:
                    expected_metric_names = metric_names
                elif metric_names != expected_metric_names:
                    raise ValueError(
                        f"method {method!r} threshold {threshold!r} prompt "
                        f"{prompt_id!r} changed the metric names"
                    )
                values: dict[str, float] = {}
                for raw_name, raw_value in raw_metrics.items():
                    name = str(raw_name)
                    if isinstance(raw_value, bool):
                        raise ValueError(f"metric {name!r} must be numeric, not boolean")
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"metric {name!r} is not numeric") from error
                    if not math.isfinite(value):
                        raise ValueError(f"metric {name!r} is not finite")
                    values[name] = value
                threshold_prompts[prompt_id] = values
            method_curve[threshold] = threshold_prompts
        normalized[method] = method_curve

    assert expected_thresholds is not None
    assert expected_prompt_ids is not None
    assert expected_metric_names is not None
    return tuple(
        {
            "method": method,
            "threshold": threshold,
            "prompt_id": prompt_id,
            **{
                metric: normalized[method][threshold][prompt_id][metric]
                for metric in sorted(expected_metric_names)
            },
        }
        for method in methods
        for threshold in sorted(expected_thresholds, key=float)
        for prompt_id in sorted(expected_prompt_ids)
    )
