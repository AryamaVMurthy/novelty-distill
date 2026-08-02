import pytest

from novelty_distill.evaluation.student_evaluation import (
    evaluate_joint_embeddings,
    summarize_generation_diagnostics,
    summarize_quality_dimensions,
    summarize_student_prompt,
)


def test_student_prompt_summary_uses_joint_clusters_and_quality() -> None:
    summary = summarize_student_prompt(
        teacher_clusters=("a", "a", "b", "c"),
        student_clusters=("a", "b", "b", "d"),
        student_quality_scores=(0.8, 0.6, 1.0, 0.4),
        student_feasibility_scores=(5, 3, 4, 2),
        nearest_training_target_similarities=(0.9, 0.7, 0.8, 0.6),
    )

    assert summary == {
        "teacher_semantic_clusters": 3,
        "student_semantic_clusters": 3,
        "teacher_mode_recall": pytest.approx(2 / 3),
        "teacher_mode_precision": pytest.approx(2 / 3),
        "cluster_jsd": pytest.approx(0.31127812445913283),
        "student_quality_mean": pytest.approx(0.7),
        "student_feasibility_mean": pytest.approx(3.5),
        "quality_adjusted_coverage": pytest.approx(2.2),
        "nearest_training_target_similarity_mean": pytest.approx(0.75),
        "nearest_training_target_similarity_max": pytest.approx(0.9),
    }


@pytest.mark.parametrize(
    ("field", "values", "match"),
    [
        ("student_clusters", ("a",), "same non-zero length"),
        ("student_quality_scores", (0.5,), "same non-zero length"),
        ("student_feasibility_scores", (4,), "same non-zero length"),
        (
            "nearest_training_target_similarities",
            (0.5,),
            "same non-zero length",
        ),
    ],
)
def test_student_prompt_summary_rejects_misaligned_student_records(
    field: str, values: tuple[object, ...], match: str
) -> None:
    arguments = {
        "teacher_clusters": ("a", "b"),
        "student_clusters": ("a", "b"),
        "student_quality_scores": (0.5, 0.6),
        "student_feasibility_scores": (3, 4),
        "nearest_training_target_similarities": (0.7, 0.8),
    }
    arguments[field] = values

    with pytest.raises(ValueError, match=match):
        summarize_student_prompt(**arguments)


def test_joint_embedding_evaluation_shares_cluster_labels_and_finds_training_neighbor() -> None:
    summary = evaluate_joint_embeddings(
        teacher_embeddings=((1.0, 0.0), (0.0, 1.0)),
        student_embeddings=((0.99, 0.01), (-1.0, 0.0)),
        training_target_embeddings=((1.0, 0.0), (0.0, -1.0)),
        student_quality_scores=(0.8, 0.4),
        student_feasibility_scores=(5, 2),
        threshold=0.95,
    )

    assert summary["teacher_semantic_clusters"] == 2
    assert summary["student_semantic_clusters"] == 2
    assert summary["teacher_mode_recall"] == pytest.approx(0.5)
    assert summary["teacher_mode_precision"] == pytest.approx(0.5)
    assert summary["nearest_training_target_similarity_mean"] == pytest.approx(0.5, abs=1e-4)
    assert summary["nearest_training_target_similarity_max"] == pytest.approx(1.0, abs=1e-4)


def test_generation_diagnostics_expose_length_stops_and_token_distribution() -> None:
    diagnostics = summarize_generation_diagnostics(
        finish_reasons=("stop", "length", "stop", "length"),
        completion_tokens=(101, 512, 203, 512),
    )

    assert diagnostics == {
        "length_stop_rate": 0.5,
        "completion_tokens_mean": 332.0,
        "completion_tokens_median": 357.5,
        "completion_tokens_max": 512,
    }


def test_generation_diagnostics_reject_missing_usage() -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        summarize_generation_diagnostics(
            finish_reasons=("stop", "length"),
            completion_tokens=(100,),
        )


def test_quality_dimension_summary_preserves_each_rubric_axis() -> None:
    dimensions = (
        {
            "relevance": 5,
            "feasibility": 3,
            "soundness": 2,
            "clarity": 4,
            "instruction_compliance": 5,
        },
        {
            "relevance": 3,
            "feasibility": 5,
            "soundness": 4,
            "clarity": 2,
            "instruction_compliance": 5,
        },
    )

    assert summarize_quality_dimensions(dimensions) == {
        "student_relevance_mean": 4.0,
        "student_feasibility_mean": 4.0,
        "student_soundness_mean": 3.0,
        "student_clarity_mean": 3.0,
        "student_instruction_compliance_mean": 5.0,
    }
