"""Prompt-paired comparisons for corrected NoveltyBench runs."""

import math
import statistics
from collections.abc import Mapping
from typing import Any

_METRICS = ("distinct_k", "utility_k")


def _validated_scores(
    method: str, scores: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    if not scores:
        raise ValueError(f"{method} has no prompt scores")
    validated: dict[str, dict[str, float]] = {}
    for prompt_id, values in scores.items():
        if not prompt_id or not all(metric in values for metric in _METRICS):
            raise ValueError(f"{method} prompt scores are missing required metrics")
        converted = {metric: float(values[metric]) for metric in _METRICS}
        if any(not math.isfinite(value) for value in converted.values()):
            raise ValueError(f"{method} prompt scores must be finite")
        validated[str(prompt_id)] = converted
    return validated


def paired_noveltybench_contrasts(
    scores_by_method: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    baseline: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Compare methods against a baseline using the prompt as the paired unit."""

    if baseline not in scores_by_method:
        raise ValueError(f"baseline {baseline} is missing")
    validated = {
        method: _validated_scores(method, scores)
        for method, scores in scores_by_method.items()
    }
    baseline_scores = validated[baseline]
    prompt_ids = set(baseline_scores)
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for method, method_scores in validated.items():
        if method == baseline:
            continue
        if set(method_scores) != prompt_ids:
            raise ValueError(f"{method} prompt IDs differ from {baseline}")
        method_result: dict[str, dict[str, float | int]] = {}
        for metric in _METRICS:
            baseline_values = [baseline_scores[prompt_id][metric] for prompt_id in prompt_ids]
            method_values = [method_scores[prompt_id][metric] for prompt_id in prompt_ids]
            deltas = [
                method_value - baseline_value
                for method_value, baseline_value in zip(
                    method_values, baseline_values, strict=True
                )
            ]
            n_prompts = len(deltas)
            method_result[metric] = {
                "n_prompts": n_prompts,
                "baseline_mean": statistics.fmean(baseline_values),
                "method_mean": statistics.fmean(method_values),
                "mean_delta": statistics.fmean(deltas),
                "delta_stderr": (
                    statistics.stdev(deltas) / math.sqrt(n_prompts)
                    if n_prompts > 1
                    else 0.0
                ),
                "improved_prompt_fraction": sum(delta > 0 for delta in deltas)
                / n_prompts,
                "tied_prompt_fraction": sum(delta == 0 for delta in deltas) / n_prompts,
                "worsened_prompt_fraction": sum(delta < 0 for delta in deltas)
                / n_prompts,
            }
        result[method] = method_result
    return result
