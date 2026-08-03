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


def same_prompt_geometry_metrics(
    *,
    candidate: Sequence[Sequence[float]],
    teacher: Sequence[Sequence[float]],
    human: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Compute one prompt's normalized cross-source and concentration metrics."""

    normalized_candidate, dimension = _normalized_matrix(
        candidate, expected_dimension=None
    )
    normalized_teacher, _ = _normalized_matrix(
        teacher, expected_dimension=dimension
    )
    normalized_human, _ = _normalized_matrix(human, expected_dimension=dimension)
    teacher_cosine = _cross_cosine(normalized_candidate, normalized_teacher)
    human_cosine = _cross_cosine(normalized_candidate, normalized_human)
    return {
        "teacher_cosine_mean": teacher_cosine,
        "human_cosine_mean": human_cosine,
        "teacher_minus_human_affinity": teacher_cosine - human_cosine,
        "within_method_pair_cosine": _within_cosine(normalized_candidate),
        "teacher_human_cosine": _cross_cosine(normalized_teacher, normalized_human),
    }


def summarize_same_prompt_geometry_metrics(
    prompt_metrics_by_method: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    samples_per_prompt: Mapping[str, Sequence[int]],
    reference_by_prompt: Mapping[str, float],
    human_method: str,
    teacher_method: str,
    base_method: str,
    embedding_dimension: int,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
    bootstrap_batch_size: int = 500,
) -> dict[str, Any]:
    """Summarize precomputed prompt metrics with bounded-memory paired bootstrap."""

    if resamples <= 0 or bootstrap_batch_size <= 0:
        raise ValueError("bootstrap resamples and batch size must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence level must lie strictly between zero and one")
    if embedding_dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    if base_method not in prompt_metrics_by_method:
        raise ValueError("same-prompt geometry requires a non-reference base method")
    if set(samples_per_prompt) != set(prompt_metrics_by_method):
        raise ValueError("sample-count methods disagree with prompt metric methods")

    prompt_ids = tuple(sorted(reference_by_prompt))
    if not prompt_ids or any(
        set(values) != set(prompt_ids) for values in prompt_metrics_by_method.values()
    ):
        raise ValueError("all geometry methods must use the same prompt population")
    if any(
        not np.isfinite(float(value))
        for value in reference_by_prompt.values()
    ):
        raise ValueError("reference geometry metrics must be finite")

    raw_by_method: dict[str, np.ndarray] = {}
    for method, prompt_metrics in prompt_metrics_by_method.items():
        raw = np.asarray(
            [
                [float(prompt_metrics[prompt_id][name]) for name in _RAW_METRICS]
                for prompt_id in prompt_ids
            ],
            dtype=np.float64,
        )
        if not np.isfinite(raw).all():
            raise ValueError("same-prompt geometry metrics must be finite")
        raw_by_method[method] = raw

    base = raw_by_method[base_method]
    extended_by_method = {
        method: np.concatenate((raw, raw - base), axis=1)
        for method, raw in raw_by_method.items()
    }
    metric_names = (*_RAW_METRICS, *_DELTA_METRICS)
    alpha = (1 - confidence_level) / 2
    rng = np.random.default_rng(seed)

    methods: dict[str, Any] = {}
    for method, values in extended_by_method.items():
        point = values.mean(axis=0)
        sampled_chunks: list[np.ndarray] = []
        for start in range(0, resamples, bootstrap_batch_size):
            current = min(bootstrap_batch_size, resamples - start)
            indices = rng.integers(
                0, len(prompt_ids), size=(current, len(prompt_ids))
            )
            sampled_chunks.append(values[indices].mean(axis=1))
        sampled = np.concatenate(sampled_chunks, axis=0)
        methods[method] = {
            "num_samples_per_prompt": sorted(set(samples_per_prompt[method])),
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
        "embedding_dimension": embedding_dimension,
        "reference": {
            "teacher_human_cosine_mean": float(
                np.mean(tuple(reference_by_prompt[prompt_id] for prompt_id in prompt_ids))
            ),
        },
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "confidence_level": confidence_level,
            "unit": "prompt",
        },
        "methods": methods,
    }


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

    prompt_metrics_by_method: dict[str, dict[str, dict[str, float]]] = {}
    samples_per_prompt: dict[str, list[int]] = {}
    reference_by_prompt: dict[str, float] = {}
    dimension: int | None = None
    for method in candidate_methods:
        prompt_metrics: dict[str, dict[str, float]] = {}
        samples_per_prompt[method] = []
        for prompt_id in prompt_ids:
            candidate = vectors[method][prompt_id]
            metrics = same_prompt_geometry_metrics(
                candidate=candidate,
                teacher=vectors[teacher_method][prompt_id],
                human=vectors[human_method][prompt_id],
            )
            prompt_metrics[prompt_id] = {
                name: metrics[name] for name in _RAW_METRICS
            }
            samples_per_prompt[method].append(len(candidate))
            if method == base_method:
                reference_by_prompt[prompt_id] = metrics["teacher_human_cosine"]
            observed = np.asarray(candidate)
            if observed.ndim == 2 and observed.shape[1] > 0:
                dimension = int(observed.shape[1])
        prompt_metrics_by_method[method] = prompt_metrics
    if dimension is None:
        raise ValueError("same-prompt geometry has no embedding dimension")
    return summarize_same_prompt_geometry_metrics(
        prompt_metrics_by_method,
        samples_per_prompt=samples_per_prompt,
        reference_by_prompt=reference_by_prompt,
        human_method=human_method,
        teacher_method=teacher_method,
        base_method=base_method,
        embedding_dimension=dimension,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
