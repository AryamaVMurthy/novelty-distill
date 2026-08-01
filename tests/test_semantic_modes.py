import pytest

from novelty_distill.evaluation.semantic_modes import mode_metrics, quality_adjusted_coverage


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
