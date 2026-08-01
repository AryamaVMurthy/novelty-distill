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


def _kl(distribution: dict[str, float], reference: dict[str, float]) -> float:
    return sum(
        probability * math.log2(probability / reference[label])
        for label, probability in distribution.items()
        if probability > 0
    )
