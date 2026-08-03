"""Metrics over externally assigned semantic idea-mode labels."""

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeMetrics:
    mode_recall: float
    mode_precision: float
    cluster_jsd: float


def mode_metrics(
    *, teacher_clusters: Iterable[str], student_clusters: Iterable[str]
) -> ModeMetrics:
    """Calculate mode-set retention and distributional Jensen-Shannon divergence."""

    teacher = tuple(teacher_clusters)
    student = tuple(student_clusters)
    if not teacher or not student:
        raise ValueError("teacher and student cluster samples must both be non-empty")
    teacher_counts = Counter(teacher)
    student_counts = Counter(student)
    shared = teacher_counts.keys() & student_counts.keys()
    recall = len(shared) / len(teacher_counts)
    precision = len(shared) / len(student_counts)
    labels = teacher_counts.keys() | student_counts.keys()
    teacher_total = len(teacher)
    student_total = len(student)
    teacher_probabilities = {
        label: teacher_counts[label] / teacher_total for label in labels
    }
    student_probabilities = {
        label: student_counts[label] / student_total for label in labels
    }
    midpoint = {
        label: (teacher_probabilities[label] + student_probabilities[label]) / 2
        for label in labels
    }
    jsd = 0.5 * _kl(teacher_probabilities, midpoint) + 0.5 * _kl(
        student_probabilities, midpoint
    )
    return ModeMetrics(mode_recall=recall, mode_precision=precision, cluster_jsd=jsd)


def quality_adjusted_coverage(
    *, clusters: Iterable[str], quality_scores: Iterable[float]
) -> float:
    """Sum the best quality observed in every recovered student mode."""

    cluster_tuple = tuple(clusters)
    score_tuple = tuple(quality_scores)
    if not cluster_tuple or len(cluster_tuple) != len(score_tuple):
        raise ValueError("clusters and quality scores must have the same non-zero length")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in score_tuple):
        raise ValueError("quality scores must be finite values in [0, 1]")
    best_by_cluster: dict[str, float] = {}
    for cluster, score in zip(cluster_tuple, score_tuple, strict=True):
        best_by_cluster[cluster] = max(best_by_cluster.get(cluster, 0), score)
    return sum(best_by_cluster.values())


def viable_semantic_yield(
    *,
    embeddings: Iterable[Iterable[float]],
    eligible: Iterable[bool],
    similarity_threshold: float,
) -> int:
    """Count the largest mutually distinct subset of individually viable responses."""

    rows = tuple(tuple(float(value) for value in row) for row in embeddings)
    flags = tuple(bool(value) for value in eligible)
    if not rows or len(rows) != len(flags):
        raise ValueError("embeddings and eligibility flags must have the same non-zero length")
    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity threshold must be between -1 and 1")
    dimensions = {len(row) for row in rows}
    if dimensions == {0} or len(dimensions) != 1:
        raise ValueError("embeddings must share one non-zero dimension")
    normalized: list[tuple[float, ...]] = []
    for row in rows:
        if any(not math.isfinite(value) for value in row):
            raise ValueError("embeddings must contain only finite values")
        norm = math.sqrt(sum(value * value for value in row))
        if norm == 0:
            raise ValueError("embeddings must be non-zero")
        normalized.append(tuple(value / norm for value in row))

    candidates = tuple(index for index, flag in enumerate(flags) if flag)
    if not candidates:
        return 0
    adjacency = [0] * len(candidates)
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            similarity = sum(
                a * b
                for a, b in zip(
                    normalized[candidates[left]], normalized[candidates[right]], strict=True
                )
            )
            if similarity < similarity_threshold:
                adjacency[left] |= 1 << right
                adjacency[right] |= 1 << left

    best = 0

    def expand(available: int, size: int) -> None:
        nonlocal best
        if size + available.bit_count() <= best:
            return
        while available:
            vertex_bit = available & -available
            vertex = vertex_bit.bit_length() - 1
            available ^= vertex_bit
            expand(available & adjacency[vertex], size + 1)
            if size + available.bit_count() <= best:
                break
        best = max(best, size)

    expand((1 << len(candidates)) - 1, 0)
    return best


def _kl(distribution: dict[str, float], reference: dict[str, float]) -> float:
    return sum(
        probability * math.log2(probability / reference[label])
        for label, probability in distribution.items()
        if probability > 0
    )
