import pytest

from novelty_distill.data.teacher_views import TeacherGeneration
from novelty_distill.evaluation.calibration_analysis import summarize_calibration_prompt
from novelty_distill.generation.sglang import GenerationRecord


def test_calibration_prompt_summary_separates_quality_modes_and_length() -> None:
    generation_records = tuple(
        GenerationRecord(
            prompt_id="p1",
            sample_index=index,
            text=f"idea-{index}",
            finish_reason="length" if index == 0 else "stop",
            model="teacher",
            request_id=f"request-{index}",
            config_hash="hash",
            prompt_tokens=10,
            completion_tokens=100 + index,
        )
        for index in range(4)
    )
    clusters = ("a", "a", "b", "c")
    qualities = (0.2, 0.8, 0.5, 0.1)
    clustered = tuple(
        TeacherGeneration(
            prompt_id="p1",
            sample_index=index,
            text=f"idea-{index}",
            quality_score=qualities[index],
            cluster_id=clusters[index],
        )
        for index in range(4)
    )
    diagnostics = {
        "primary_embedding": "instructed",
        "raw": {
            "cosine_min": 0.5,
            "cosine_mean": 0.7,
            "cosine_max": 0.9,
            "clusters_by_threshold": {"0.700": 1, "0.820": 2},
        },
        "instructed": {
            "cosine_min": 0.4,
            "cosine_mean": 0.6,
            "cosine_max": 0.8,
            "clusters_by_threshold": {"0.700": 2, "0.820": 3},
        },
    }

    summary = summarize_calibration_prompt(
        generation_records=generation_records,
        clustered_records=clustered,
        diagnostics=diagnostics,
    )

    assert summary["prompt_id"] == "p1"
    assert summary["unique_text_rate"] == 1
    assert summary["length_termination_rate"] == pytest.approx(0.25)
    assert summary["quality_mean"] == pytest.approx(0.4)
    assert summary["semantic_clusters"] == 3
    assert summary["quality_adjusted_coverage"] == pytest.approx(1.4)
    assert summary["instructed_clusters_by_threshold"]["0.820"] == 3


def test_calibration_prompt_summary_rejects_misaligned_records() -> None:
    generation = GenerationRecord(
        prompt_id="p1",
        sample_index=0,
        text="idea",
        finish_reason="stop",
        model="teacher",
        request_id="request",
        config_hash="hash",
    )
    clustered = TeacherGeneration(
        prompt_id="p1",
        sample_index=1,
        text="other",
        quality_score=0.5,
        cluster_id="a",
    )

    with pytest.raises(ValueError, match="align"):
        summarize_calibration_prompt(
            generation_records=(generation,),
            clustered_records=(clustered,),
            diagnostics={},
        )
