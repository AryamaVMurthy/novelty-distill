import pytest

from novelty_distill.evaluation.semantic_validity import SemanticValidityPolicy


def test_uncalibrated_single_threshold_metrics_cannot_be_primary() -> None:
    policy = SemanticValidityPolicy(
        name="test-v1",
        single_threshold_inference_allowed=False,
        full_threshold_curve_required=True,
        claim_boundary="descriptive only",
    )

    with pytest.raises(ValueError, match="teacher_mode_recall.*single-threshold"):
        policy.validate_analysis_config(
            {
                "metrics": ["student_soundness_mean", "teacher_mode_recall"],
                "threshold_metric_directions": {"teacher_mode_recall": "higher"},
            }
        )


def test_quality_only_primary_metrics_pass_uncalibrated_policy() -> None:
    policy = SemanticValidityPolicy(
        name="test-v1",
        single_threshold_inference_allowed=False,
        full_threshold_curve_required=True,
        claim_boundary="descriptive only",
    )

    policy.validate_analysis_config(
        {
            "metrics": ["student_feasibility_mean", "student_soundness_mean"],
            "threshold_metric_directions": {"teacher_mode_recall": "higher"},
        }
    )
