"""Prompt-level metrics comparing student samples with jointly clustered teacher samples."""

import math
import statistics
from collections.abc import Iterable
from typing import Any

from novelty_distill.evaluation.semantic_modes import (
    mode_metrics,
    quality_adjusted_coverage,
)
from novelty_distill.evaluation.teacher_annotation import cluster_cosine_embeddings


def evaluate_joint_embeddings(
    *,
    teacher_embeddings: Iterable[Iterable[float]],
    student_embeddings: Iterable[Iterable[float]],
    training_target_embeddings: Iterable[Iterable[float]],
    student_quality_scores: Iterable[float],
    student_feasibility_scores: Iterable[int],
    threshold: float,
) -> dict[str, Any]:
    """Cluster teacher/student samples jointly and compute training-set proximity."""

    teacher = tuple(tuple(float(value) for value in row) for row in teacher_embeddings)
    student = tuple(tuple(float(value) for value in row) for row in student_embeddings)
    targets = tuple(tuple(float(value) for value in row) for row in training_target_embeddings)
    if not teacher or not student or not targets:
        raise ValueError("teacher, student, and training embeddings must be non-empty")
    normalized_teacher = _normalize(teacher)
    normalized_student = _normalize(student)
    normalized_targets = _normalize(targets)
    dimensions = {
        len(row) for row in (*normalized_teacher, *normalized_student, *normalized_targets)
    }
    if len(dimensions) != 1:
        raise ValueError("all embeddings must share one dimension")
    labels = cluster_cosine_embeddings(
        (*normalized_teacher, *normalized_student), threshold=threshold
    )
    nearest = tuple(
        max(_dot(embedding, target) for target in normalized_targets)
        for embedding in normalized_student
    )
    return summarize_student_prompt(
        teacher_clusters=labels[: len(normalized_teacher)],
        student_clusters=labels[len(normalized_teacher) :],
        student_quality_scores=student_quality_scores,
        student_feasibility_scores=student_feasibility_scores,
        nearest_training_target_similarities=nearest,
    )


def summarize_student_prompt(
    *,
    teacher_clusters: Iterable[str],
    student_clusters: Iterable[str],
    student_quality_scores: Iterable[float],
    student_feasibility_scores: Iterable[int],
    nearest_training_target_similarities: Iterable[float],
) -> dict[str, Any]:
    """Summarize one prompt after assigning teacher and student joint cluster labels."""

    teacher = tuple(teacher_clusters)
    student = tuple(student_clusters)
    qualities = tuple(float(value) for value in student_quality_scores)
    feasibility = tuple(int(value) for value in student_feasibility_scores)
    nearest = tuple(float(value) for value in nearest_training_target_similarities)
    if not teacher:
        raise ValueError("teacher clusters must be non-empty")
    student_lengths = {len(student), len(qualities), len(feasibility), len(nearest)}
    if len(student_lengths) != 1 or not student:
        raise ValueError("student metrics must have the same non-zero length")
    if any(not cluster for cluster in (*teacher, *student)):
        raise ValueError("cluster labels must be non-empty")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in qualities):
        raise ValueError("student quality scores must be finite values in [0, 1]")
    if any(not 1 <= value <= 5 for value in feasibility):
        raise ValueError("student feasibility scores must be integers in [1, 5]")
    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in nearest):
        raise ValueError("nearest similarities must be finite values in [-1, 1]")

    modes = mode_metrics(teacher_clusters=teacher, student_clusters=student)
    return {
        "teacher_semantic_clusters": len(set(teacher)),
        "student_semantic_clusters": len(set(student)),
        "teacher_mode_recall": modes.mode_recall,
        "teacher_mode_precision": modes.mode_precision,
        "cluster_jsd": modes.cluster_jsd,
        "student_quality_mean": statistics.fmean(qualities),
        "student_feasibility_mean": statistics.fmean(feasibility),
        "quality_adjusted_coverage": quality_adjusted_coverage(
            clusters=student, quality_scores=qualities
        ),
        "nearest_training_target_similarity_mean": statistics.fmean(nearest),
        "nearest_training_target_similarity_max": max(nearest),
    }


def summarize_generation_diagnostics(
    *,
    finish_reasons: Iterable[str],
    completion_tokens: Iterable[int],
) -> dict[str, float | int]:
    """Expose truncation and response-length confounds for one prompt."""

    reasons = tuple(str(value) for value in finish_reasons)
    tokens = tuple(int(value) for value in completion_tokens)
    if not reasons or len(reasons) != len(tokens):
        raise ValueError("finish reasons and token counts must have the same non-zero length")
    if any(not reason for reason in reasons):
        raise ValueError("finish reasons must be non-empty")
    if any(value < 0 for value in tokens):
        raise ValueError("completion token counts must be non-negative")
    return {
        "length_stop_rate": sum(reason == "length" for reason in reasons) / len(reasons),
        "completion_tokens_mean": statistics.fmean(tokens),
        "completion_tokens_median": statistics.median(tokens),
        "completion_tokens_max": max(tokens),
    }


def _normalize(rows: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    normalized = []
    for row in rows:
        if not row or any(not math.isfinite(value) for value in row):
            raise ValueError("embeddings must contain finite non-empty rows")
        norm = math.sqrt(sum(value * value for value in row))
        if norm == 0:
            raise ValueError("embeddings must be non-zero")
        normalized.append(tuple(value / norm for value in row))
    return tuple(normalized)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
