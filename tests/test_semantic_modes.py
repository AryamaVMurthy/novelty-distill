import pytest

from novelty_distill.evaluation.semantic_modes import (
    mode_metrics,
    quality_adjusted_coverage,
    viable_semantic_yield,
)


def test_mode_metrics_distinguish_overlap_from_distribution_shift() -> None:
    metrics = mode_metrics(
        teacher_clusters=("a", "a", "b", "c"),
        student_clusters=("a", "b", "b", "d"),
    )

    assert metrics.mode_recall == pytest.approx(2 / 3)
    assert metrics.mode_precision == pytest.approx(2 / 3)
    assert 0 < metrics.cluster_jsd < 1

    disjoint = mode_metrics(teacher_clusters=("a", "a"), student_clusters=("b", "b"))
    assert disjoint.mode_recall == 0
    assert disjoint.mode_precision == 0
    assert disjoint.cluster_jsd == pytest.approx(1.0)


def test_quality_adjusted_coverage_counts_only_best_output_per_mode() -> None:
    score = quality_adjusted_coverage(
        clusters=("a", "a", "b", "c"),
        quality_scores=(0.2, 0.8, 0.5, 0.1),
    )

    assert score == pytest.approx(1.4)


def test_viable_semantic_yield_counts_largest_mutually_distinct_set() -> None:
    result = viable_semantic_yield(
        embeddings=((1.0, 0.0), (0.99, 0.01), (0.0, 1.0)),
        eligible=(True, True, True),
        similarity_threshold=0.95,
    )

    assert result == 2


def test_viable_semantic_yield_is_zero_without_viable_responses() -> None:
    assert (
        viable_semantic_yield(
            embeddings=((1.0, 0.0), (0.0, 1.0)),
            eligible=(False, False),
            similarity_threshold=0.95,
        )
        == 0
    )
