import pytest

from novelty_distill.evaluation.score_analysis import (
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
    markdown = render_score_summary_markdown(summary)
    assert "# Scored generation diagnostics" in markdown
    assert "not a novelty score" in markdown
    assert "| `clarity` |" in markdown


def test_summarize_score_payloads_rejects_mixed_judges() -> None:
    first = _payload("p1", (5, 3))
    second = _payload("p2", (4, 4))
    second["judge"] = {"model": "other", "revision": "b" * 40, "max_tokens": 128}

    with pytest.raises(ValueError, match="judge"):
        summarize_score_payloads((first, second))
