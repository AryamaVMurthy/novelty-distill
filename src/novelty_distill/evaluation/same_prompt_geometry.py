"""Prompt-matched teacher/human embedding geometry as a secondary diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_RAW_METRICS = (
    "teacher_cosine_mean",
    "human_cosine_mean",
    "teacher_minus_human_affinity",
    "within_method_pair_cosine",
)
_DELTA_METRICS = (
    "teacher_cosine_delta_vs_base",
    "human_cosine_delta_vs_base",
    "teacher_minus_human_affinity_delta_vs_base",
    "within_method_pair_cosine_delta_vs_base",
)


def _normalized_matrix(
    values: Sequence[Sequence[float]], *, expected_dimension: int | None
) -> tuple[np.ndarray, int]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embedding groups must be non-empty two-dimensional matrices")
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise ValueError("all embedding vectors must share one dimension")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding vectors must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must be non-zero")
    return matrix / norms, int(matrix.shape[1])


def _cross_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(left @ right.T))


def _within_cosine(values: np.ndarray) -> float:
    if len(values) < 2:
        raise ValueError("same-prompt geometry candidates require at least two samples")
    similarities = values @ values.T
    upper = similarities[np.triu_indices(len(values), k=1)]
    return float(np.mean(upper))


def analyze_same_prompt_geometry(
    vectors: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
    *,
    human_method: str,
    teacher_method: str,
    base_method: str,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compare each method with prompt-matched teacher and human embedding sets.

    Every prompt is one bootstrap unit. Positive ``teacher_minus_human_affinity``
    means that a method is closer to the teacher outputs than to the historical
    human idea under the frozen embedding representation.
    """

    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence level must lie strictly between zero and one")
    required = {human_method, teacher_method, base_method}
    if missing := sorted(required - set(vectors)):
        raise ValueError(f"same-prompt geometry is missing required methods: {missing}")
    candidate_methods = tuple(
        method for method in vectors if method not in {human_method, teacher_method}
    )
    if not candidate_methods or base_method not in candidate_methods:
        raise ValueError("same-prompt geometry requires a non-reference base method")

    prompt_ids = tuple(sorted(vectors[human_method]))
    if not prompt_ids or any(set(values) != set(prompt_ids) for values in vectors.values()):
        raise ValueError("all geometry methods must use the same prompt population")

    normalized: dict[str, dict[str, np.ndarray]] = {}
    dimension: int | None = None
    for method, prompt_values in vectors.items():
        normalized[method] = {}
        for prompt_id in prompt_ids:
            matrix, observed_dimension = _normalized_matrix(
                prompt_values[prompt_id], expected_dimension=dimension
            )
            dimension = observed_dimension
            normalized[method][prompt_id] = matrix

    reference_values: list[float] = []
    raw_by_method: dict[str, np.ndarray] = {}
    prompt_metrics_by_method: dict[str, dict[str, dict[str, float]]] = {}
    for method in candidate_methods:
        rows: list[tuple[float, float, float, float]] = []
        prompt_metrics: dict[str, dict[str, float]] = {}
        for prompt_id in prompt_ids:
            candidate = normalized[method][prompt_id]
            teacher = normalized[teacher_method][prompt_id]
            human = normalized[human_method][prompt_id]
            teacher_cosine = _cross_cosine(candidate, teacher)
            human_cosine = _cross_cosine(candidate, human)
            within_cosine = _within_cosine(candidate)
            row = (
                teacher_cosine,
                human_cosine,
                teacher_cosine - human_cosine,
                within_cosine,
            )
            rows.append(row)
            prompt_metrics[prompt_id] = dict(zip(_RAW_METRICS, row, strict=True))
            if method == base_method:
                reference_values.append(_cross_cosine(teacher, human))
        raw_by_method[method] = np.asarray(rows, dtype=np.float64)
        prompt_metrics_by_method[method] = prompt_metrics

    base = raw_by_method[base_method]
    extended_by_method = {
        method: np.concatenate((raw, raw - base), axis=1)
        for method, raw in raw_by_method.items()
    }
    metric_names = (*_RAW_METRICS, *_DELTA_METRICS)
    alpha = (1 - confidence_level) / 2
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(prompt_ids), size=(resamples, len(prompt_ids)))

    methods: dict[str, Any] = {}
    for method, values in extended_by_method.items():
        point = values.mean(axis=0)
        sampled = values[indices].mean(axis=1)
        methods[method] = {
            "num_samples_per_prompt": sorted(
                {len(normalized[method][prompt_id]) for prompt_id in prompt_ids}
            ),
            "metrics": {
                name: {
                    "estimate": float(point[column]),
                    "ci_low": float(np.quantile(sampled[:, column], alpha)),
                    "ci_high": float(np.quantile(sampled[:, column], 1 - alpha)),
                }
                for column, name in enumerate(metric_names)
            },
            "prompt_metrics": prompt_metrics_by_method[method],
        }

    return {
        "status": "secondary_descriptive",
        "num_prompts": len(prompt_ids),
        "human_method": human_method,
        "teacher_method": teacher_method,
        "base_method": base_method,
        "embedding_dimension": dimension,
        "reference": {
            "teacher_human_cosine_mean": float(np.mean(reference_values)),
        },
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "confidence_level": confidence_level,
            "unit": "prompt",
        },
        "methods": methods,
    }
