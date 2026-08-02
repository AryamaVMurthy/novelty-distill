"""Preregistered prompt-paired contrast analysis across an evaluation matrix."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from novelty_distill.evaluation.statistics import holm_adjust, paired_bootstrap


def analyze_contrasts(
    *,
    rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, str]],
    metrics: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Analyze treatment-minus-reference effects with per-metric Holm correction."""

    if not rows or not contrasts or not metrics:
        raise ValueError("rows, contrasts, and metrics must be non-empty")
    if len(set(metrics)) != len(metrics):
        raise ValueError("metric names must be unique")
    by_method: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        prompt_id = str(row.get("prompt_id", ""))
        if not method or not prompt_id:
            raise ValueError("every row needs non-empty method and prompt_id")
        prompt_rows = by_method.setdefault(method, {})
        if prompt_id in prompt_rows:
            raise ValueError(f"duplicate prompt {prompt_id!r} for method {method!r}")
        prompt_rows[prompt_id] = row

    contrast_ids = [str(contrast.get("id", "")) for contrast in contrasts]
    if any(not contrast_id for contrast_id in contrast_ids) or len(contrast_ids) != len(
        set(contrast_ids)
    ):
        raise ValueError("contrast IDs must be non-empty and unique")

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in metrics:
        metric_results: dict[str, dict[str, Any]] = {}
        raw_p_values: dict[str, float] = {}
        for contrast in contrasts:
            contrast_id = str(contrast["id"])
            reference_name = str(contrast["reference"])
            treatment_name = str(contrast["treatment"])
            if reference_name not in by_method or treatment_name not in by_method:
                raise ValueError(f"contrast {contrast_id!r} references an absent method")
            reference = by_method[reference_name]
            treatment = by_method[treatment_name]
            if reference.keys() != treatment.keys():
                raise ValueError(f"contrast {contrast_id!r} does not have exactly paired prompts")
            prompt_ids = sorted(reference)
            try:
                reference_values = tuple(float(reference[prompt][metric]) for prompt in prompt_ids)
                treatment_values = tuple(float(treatment[prompt][metric]) for prompt in prompt_ids)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"contrast {contrast_id!r} has invalid metric {metric!r}"
                ) from error
            if any(not math.isfinite(value) for value in (*reference_values, *treatment_values)):
                raise ValueError(f"contrast {contrast_id!r} has non-finite metric {metric!r}")
            estimate = paired_bootstrap(
                reference=reference_values,
                treatment=treatment_values,
                samples=bootstrap_samples,
                seed=seed,
            )
            serialized = asdict(estimate)
            if not math.isfinite(serialized["effect_size"]):
                serialized["effect_size"] = None
            serialized.update(
                {"reference": reference_name, "treatment": treatment_name}
            )
            metric_results[contrast_id] = serialized
            raw_p_values[contrast_id] = estimate.p_value
        for contrast_id, adjusted in holm_adjust(raw_p_values).items():
            metric_results[contrast_id]["holm_p_value"] = adjusted
        results[metric] = metric_results
    return results
