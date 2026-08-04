import pytest

from novelty_distill.evaluation.score_analysis import (
    evaluate_score_discrimination,
    render_score_summary_markdown,
    summarize_score_payloads,
)


def _payload(prompt_id: str, ratings: tuple[int, int]) -> dict[str, object]:
    records = []
    for index, rating in enumerate(ratings):
        dimensions = {
            "relevance": rating,
            "feasibility": rating,
            "soundness": rating,
            "clarity": rating,
            "instruction_compliance": rating,
        }
        records.append(
            {
                "prompt_id": prompt_id,
                "sample_index": index,
                "quality_score": (rating - 1) / 4,
                "dimensions": dimensions,
                "finish_reason": "length" if index else "stop",
                "completion_tokens": 10 + index,
            }
        )
    return {
        "judge": {"model": "judge", "revision": "a" * 40, "max_tokens": 128},
        "prompt_id": prompt_id,
        "records": records,
    }


def test_summarize_score_payloads_exposes_ceiling_and_within_prompt_spread() -> None:
    summary = summarize_score_payloads((_payload("p1", (5, 3)), _payload("p2", (4, 4))))

    assert summary["num_prompts"] == 2
    assert summary["num_samples"] == 4
    assert summary["quality"]["mean"] == pytest.approx(0.75)
    assert summary["quality"]["within_prompt_range_mean"] == pytest.approx(0.25)
    assert summary["dimensions"]["clarity"]["ceiling_rate"] == pytest.approx(0.25)
    assert summary["generation"]["length_stop_rate"] == pytest.approx(0.5)
    assert summary["generation"]["completion_tokens_median"] == pytest.approx(10.5)
    assert summary["generation"]["within_prompt_quality_length_pearson"] == pytest.approx(
        -(0.5**0.5)
    )
    markdown = render_score_summary_markdown(summary)
    assert "# Scored generation diagnostics" in markdown
    assert "not a novelty score" in markdown
    assert "| `clarity` |" in markdown


def test_discrimination_gate_keeps_only_non_saturated_scientific_axes_primary() -> None:
    summary = summarize_score_payloads((_payload("p1", (5, 3)), _payload("p2", (4, 4))))
    policy = {
        "schema_version": 1,
        "policy": "judge-discrimination-v1",
        "core_dimensions": ["feasibility", "soundness"],
        "quality_ceiling_rate_max": 0.5,
        "core_dimension_ceiling_rate_max": 0.8,
        "within_prompt_unique_scores_mean_min": 1.5,
        "metric_status": {
            "student_feasibility_mean": "primary_eligible",
            "student_soundness_mean": "primary_eligible",
            "student_quality_mean": "diagnostic_only",
        },
        "claim_boundary": "Operational quality only.",
    }

    result = evaluate_score_discrimination(summary, policy)

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["metric_status"]["student_quality_mean"] == "diagnostic_only"


def test_discrimination_gate_fails_closed_when_core_axis_is_saturated() -> None:
    summary = summarize_score_payloads((_payload("p1", (5, 5)), _payload("p2", (5, 5))))
    policy = {
        "schema_version": 1,
        "policy": "judge-discrimination-v1",
        "core_dimensions": ["feasibility", "soundness"],
        "quality_ceiling_rate_max": 0.5,
        "core_dimension_ceiling_rate_max": 0.8,
        "within_prompt_unique_scores_mean_min": 1.5,
        "metric_status": {},
        "claim_boundary": "Operational quality only.",
    }

    result = evaluate_score_discrimination(summary, policy)

    assert result["passed"] is False
    assert any("feasibility" in failure for failure in result["failures"])
    assert any("within_prompt_unique_scores_mean" in failure for failure in result["failures"])


def test_summarize_score_payloads_rejects_mixed_judges() -> None:
    first = _payload("p1", (5, 3))
    second = _payload("p2", (4, 4))
    second["judge"] = {"model": "other", "revision": "b" * 40, "max_tokens": 128}

    with pytest.raises(ValueError, match="judge"):
        summarize_score_payloads((first, second))
