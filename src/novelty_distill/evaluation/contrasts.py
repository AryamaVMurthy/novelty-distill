"""Preregistered prompt-paired contrast analysis across an evaluation matrix."""

import math
import statistics
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


def analyze_threshold_directions(
    *,
    rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, str]],
    metric_directions: Mapping[str, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Count favorable, tied, and unfavorable paired prompts over each threshold."""

    if not rows or not contrasts or not metric_directions:
        raise ValueError("rows, contrasts, and metric directions must be non-empty")
    if any(direction not in {"higher", "lower"} for direction in metric_directions.values()):
        raise ValueError("metric directions must be either 'higher' or 'lower'")

    by_method: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        threshold = str(row.get("threshold", ""))
        prompt_id = str(row.get("prompt_id", ""))
        if not method or not threshold or not prompt_id:
            raise ValueError("every threshold row needs method, threshold, and prompt_id")
        try:
            numeric_threshold = float(threshold)
        except ValueError as error:
            raise ValueError(f"threshold {threshold!r} is not numeric") from error
        if not math.isfinite(numeric_threshold):
            raise ValueError(f"threshold {threshold!r} is not finite")
        prompt_rows = by_method.setdefault(method, {}).setdefault(threshold, {})
        if prompt_id in prompt_rows:
            raise ValueError(
                f"duplicate prompt {prompt_id!r} for method {method!r} threshold {threshold!r}"
            )
        prompt_rows[prompt_id] = row

    contrast_ids = [str(contrast.get("id", "")) for contrast in contrasts]
    if any(not contrast_id for contrast_id in contrast_ids) or len(contrast_ids) != len(
        set(contrast_ids)
    ):
        raise ValueError("contrast IDs must be non-empty and unique")

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for metric, direction in metric_directions.items():
        metric_results: dict[str, dict[str, Any]] = {}
        multiplier = 1.0 if direction == "higher" else -1.0
        for contrast in contrasts:
            contrast_id = str(contrast["id"])
            reference_name = str(contrast["reference"])
            treatment_name = str(contrast["treatment"])
            if reference_name not in by_method or treatment_name not in by_method:
                raise ValueError(f"contrast {contrast_id!r} references an absent method")
            reference_curve = by_method[reference_name]
            treatment_curve = by_method[treatment_name]
            if reference_curve.keys() != treatment_curve.keys():
                raise ValueError(
                    f"contrast {contrast_id!r} does not have exactly paired thresholds"
                )

            threshold_results: dict[str, dict[str, float | int]] = {}
            mean_directions: set[str] = set()
            for threshold in sorted(reference_curve, key=float):
                reference = reference_curve[threshold]
                treatment = treatment_curve[threshold]
                if reference.keys() != treatment.keys():
                    raise ValueError(
                        f"contrast {contrast_id!r} threshold {threshold!r} does not have "
                        "exactly paired prompts"
                    )
                differences: list[float] = []
                favorable = tied = unfavorable = 0
                for prompt_id in sorted(reference):
                    try:
                        reference_value = float(reference[prompt_id][metric])
                        treatment_value = float(treatment[prompt_id][metric])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"contrast {contrast_id!r} has invalid metric {metric!r}"
                        ) from error
                    if not math.isfinite(reference_value) or not math.isfinite(treatment_value):
                        raise ValueError(
                            f"contrast {contrast_id!r} has non-finite metric {metric!r}"
                        )
                    difference = treatment_value - reference_value
                    differences.append(difference)
                    desirable_difference = multiplier * difference
                    if math.isclose(desirable_difference, 0.0, abs_tol=1e-12):
                        tied += 1
                    elif desirable_difference > 0:
                        favorable += 1
                    else:
                        unfavorable += 1
                mean_difference = statistics.fmean(differences)
                desirable_mean = multiplier * mean_difference
                if math.isclose(desirable_mean, 0.0, abs_tol=1e-12):
                    mean_directions.add("tied")
                elif desirable_mean > 0:
                    mean_directions.add("favorable")
                else:
                    mean_directions.add("unfavorable")
                count = len(differences)
                threshold_results[threshold] = {
                    "n": count,
                    "mean_difference": mean_difference,
                    "favorable_count": favorable,
                    "tied_count": tied,
                    "unfavorable_count": unfavorable,
                    "favorable_rate": favorable / count,
                    "tied_rate": tied / count,
                    "unfavorable_rate": unfavorable / count,
                }
            stable_direction = (
                next(iter(mean_directions)) if len(mean_directions) == 1 else "mixed"
            )
            metric_results[contrast_id] = {
                "reference": reference_name,
                "treatment": treatment_name,
                "desirable_direction": direction,
                "stable_mean_direction": stable_direction,
                "thresholds": threshold_results,
            }
        results[metric] = metric_results
    return results
