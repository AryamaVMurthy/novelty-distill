"""Descriptive diagnostics for the compact three-seed evaluation matrix."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

JUDGE_DIMENSIONS = (
    "student_relevance_mean",
    "student_feasibility_mean",
    "student_soundness_mean",
    "student_clarity_mean",
    "student_instruction_compliance_mean",
)


def _mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("diagnostic values must be non-empty and finite")
    return statistics.fmean(values)


def _sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must be bounded")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must have equal nonzero length")
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return (
        sum(first * second for first, second in zip(left_centered, right_centered, strict=True))
        / denominator
    )


def _distribution(values: Sequence[float], lengths: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": _mean(values),
        "prompt_population_sd": statistics.pstdev(values),
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "prompt_ceiling_rate": _mean([float(value == 5.0) for value in values]),
        "prompt_near_ceiling_rate": _mean([float(value >= 4.75) for value in values]),
        "prompt_low_rate": _mean([float(value <= 3.0) for value in values]),
        "prompt_length_pearson": _pearson(values, lengths),
    }


def _validated_payloads(
    *,
    evaluations: Mapping[str, Mapping[int, Mapping[str, Any]]],
    controls: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    semantic_metrics: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    if not evaluations or not controls or not seeds:
        raise ValueError("evaluations, controls, and seeds must be non-empty")
    expected_seeds = set(seeds)
    payloads: list[Mapping[str, Any]] = []
    for method, by_seed in evaluations.items():
        if set(by_seed) != expected_seeds:
            raise ValueError(f"{method} must contain exactly the expected seeds")
        payloads.extend(by_seed.values())
    payloads.extend(controls.values())

    first = payloads[0]
    prompt_metrics = first.get("prompt_metrics")
    thresholds = first.get("threshold_sensitivity")
    if not isinstance(prompt_metrics, dict) or not isinstance(thresholds, dict):
        raise ValueError("evaluation payloads need prompt and threshold metrics")
    prompt_ids = tuple(sorted(prompt_metrics))
    threshold_ids = tuple(sorted(thresholds, key=float))
    if not prompt_ids or not threshold_ids:
        raise ValueError("prompt and threshold metrics must be non-empty")

    for payload in payloads:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported evaluation schema")
        current_prompts = payload.get("prompt_metrics")
        current_thresholds = payload.get("threshold_sensitivity")
        if not isinstance(current_prompts, dict) or tuple(sorted(current_prompts)) != prompt_ids:
            raise ValueError("evaluation prompt ids are not aligned")
        if (
            not isinstance(current_thresholds, dict)
            or tuple(sorted(current_thresholds, key=float)) != threshold_ids
        ):
            raise ValueError("evaluation threshold grids are not aligned")
        if int(payload.get("num_prompts", -1)) != len(prompt_ids):
            raise ValueError("evaluation prompt count does not match prompt metrics")
        for prompt_id in prompt_ids:
            row = current_prompts[prompt_id]
            for metric in (*JUDGE_DIMENSIONS, "completion_tokens_mean"):
                value = float(row.get(metric, math.nan))
                if not math.isfinite(value):
                    raise ValueError(f"missing or invalid prompt metric {metric}")
        for threshold in threshold_ids:
            row = current_thresholds[threshold]
            for metric in semantic_metrics:
                value = float(row.get(metric, math.nan))
                if not math.isfinite(value):
                    raise ValueError(f"missing or invalid threshold metric {metric}")
    return prompt_ids, threshold_ids, len(prompt_ids)


def _run_judge_diagnostics(payload: Mapping[str, Any], prompt_ids: Sequence[str]) -> dict[str, Any]:
    prompt_metrics = payload["prompt_metrics"]
    lengths = [
        float(prompt_metrics[prompt_id]["completion_tokens_mean"]) for prompt_id in prompt_ids
    ]
    return {
        "dimensions": {
            metric: _distribution(
                [float(prompt_metrics[prompt_id][metric]) for prompt_id in prompt_ids], lengths
            )
            for metric in JUDGE_DIMENSIONS
        },
        "completion_tokens_mean": _mean(lengths),
        "length_stop_rate": float(payload["overall"]["length_stop_rate"]),
    }


def _aggregate_judge_runs(runs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "mean",
        "prompt_population_sd",
        "prompt_ceiling_rate",
        "prompt_near_ceiling_rate",
        "prompt_low_rate",
        "prompt_length_pearson",
    )
    dimensions: dict[str, Any] = {}
    for metric in JUDGE_DIMENSIONS:
        dimensions[metric] = {}
        for field in fields:
            values = [
                float(run["dimensions"][metric][field])
                for run in runs.values()
                if run["dimensions"][metric][field] is not None
            ]
            dimensions[metric][field] = _mean(values) if values else None
            dimensions[metric][f"{field}_seed_sd"] = _sample_sd(values) if values else None
    length_means = [float(run["completion_tokens_mean"]) for run in runs.values()]
    stop_rates = [float(run["length_stop_rate"]) for run in runs.values()]
    return {
        "dimensions": dimensions,
        "completion_tokens_mean": _mean(length_means),
        "completion_tokens_mean_seed_sd": _sample_sd(length_means),
        "length_stop_rate": _mean(stop_rates),
        "length_stop_rate_seed_sd": _sample_sd(stop_rates),
    }


def analyze_three_seed_descriptive_diagnostics(
    *,
    evaluations: Mapping[str, Mapping[int, Mapping[str, Any]]],
    controls: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    contrasts: Sequence[Mapping[str, str]],
    semantic_metrics: Mapping[str, str],
) -> dict[str, Any]:
    """Summarize judge saturation and threshold robustness without inference."""

    prompt_ids, thresholds, num_prompts = _validated_payloads(
        evaluations=evaluations,
        controls=controls,
        seeds=seeds,
        semantic_metrics=tuple(semantic_metrics),
    )
    judge_runs = {
        method: {
            str(seed): _run_judge_diagnostics(payload, prompt_ids)
            for seed, payload in sorted(by_seed.items())
        }
        for method, by_seed in evaluations.items()
    }
    judge = {
        "trained_methods": {
            method: {
                "runs": runs,
                "across_seeds": _aggregate_judge_runs(
                    {int(seed): run for seed, run in runs.items()}
                ),
            }
            for method, runs in judge_runs.items()
        },
        "fixed_controls": {
            method: _run_judge_diagnostics(payload, prompt_ids)
            for method, payload in controls.items()
        },
    }

    method_curves: dict[str, Any] = {}
    for metric in semantic_metrics:
        method_curves[metric] = {}
        for method, by_seed in evaluations.items():
            method_curves[metric][method] = {}
            for threshold in thresholds:
                per_seed = {
                    str(seed): float(by_seed[seed]["threshold_sensitivity"][threshold][metric])
                    for seed in seeds
                }
                values = list(per_seed.values())
                method_curves[metric][method][threshold] = {
                    "mean": _mean(values),
                    "seed_sd": _sample_sd(values),
                    "per_seed": per_seed,
                }
        for method, payload in controls.items():
            method_curves[metric][method] = {
                threshold: {
                    "mean": float(payload["threshold_sensitivity"][threshold][metric]),
                    "seed_sd": None,
                    "per_seed": None,
                    "fixed_control": True,
                }
                for threshold in thresholds
            }

    contrast_results: dict[str, Any] = {}
    methods = set(evaluations) | set(controls)
    for contrast in contrasts:
        contrast_id = str(contrast["id"])
        reference = str(contrast["reference"])
        treatment = str(contrast["treatment"])
        if reference not in methods or treatment not in evaluations:
            raise ValueError(f"invalid descriptive contrast {contrast_id}")
        contrast_results[contrast_id] = {}
        for metric, direction in semantic_metrics.items():
            if direction not in {"higher", "lower"}:
                raise ValueError(f"invalid direction for {metric}")
            by_threshold: dict[str, Any] = {}
            signs: list[int] = []
            for threshold in thresholds:
                per_seed: dict[str, float] = {}
                for seed in seeds:
                    treatment_value = float(
                        evaluations[treatment][seed]["threshold_sensitivity"][threshold][metric]
                    )
                    reference_payload = (
                        evaluations[reference][seed]
                        if reference in evaluations
                        else controls[reference]
                    )
                    reference_value = float(
                        reference_payload["threshold_sensitivity"][threshold][metric]
                    )
                    per_seed[str(seed)] = treatment_value - reference_value
                raw_values = list(per_seed.values())
                oriented_values = [
                    value if direction == "higher" else -value for value in raw_values
                ]
                oriented_mean = _mean(oriented_values)
                sign = 1 if oriented_mean > 1e-12 else -1 if oriented_mean < -1e-12 else 0
                signs.append(sign)
                by_threshold[threshold] = {
                    "mean_difference": _mean(raw_values),
                    "seed_sd": _sample_sd(raw_values),
                    "oriented_mean_difference": oriented_mean,
                    "per_seed_difference": per_seed,
                    "favorable_seed_count": sum(value > 1e-12 for value in oriented_values),
                    "unfavorable_seed_count": sum(value < -1e-12 for value in oriented_values),
                    "tied_seed_count": sum(abs(value) <= 1e-12 for value in oriented_values),
                }
            oriented_means = [
                float(by_threshold[threshold]["oriented_mean_difference"])
                for threshold in thresholds
            ]
            contrast_results[contrast_id][metric] = {
                "desirable_direction": direction,
                "favorable_threshold_count": signs.count(1),
                "unfavorable_threshold_count": signs.count(-1),
                "tied_threshold_count": signs.count(0),
                "stable_direction": (
                    "favorable"
                    if all(sign == 1 for sign in signs)
                    else "unfavorable"
                    if all(sign == -1 for sign in signs)
                    else "mixed"
                ),
                "oriented_delta_min": min(oriented_means),
                "oriented_delta_max": max(oriented_means),
                "thresholds": by_threshold,
            }

    return {
        "schema_version": 1,
        "num_prompts": num_prompts,
        "seeds": list(seeds),
        "thresholds": list(thresholds),
        "judge_prompt_mean_diagnostics": judge,
        "semantic_method_curves": method_curves,
        "semantic_contrast_stability": contrast_results,
        "claim_boundary": (
            "Judge distributions are automatic rubric diagnostics. Semantic curves are "
            "embedding-defined descriptive sensitivity analyses; without human boundary "
            "calibration they do not establish semantic equivalence, diversity, or novelty."
        ),
    }
