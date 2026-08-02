import pytest

from novelty_distill.evaluation.matrix import collect_prompt_metric_rows


def _evaluation(offset: float = 0.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "prompt_metrics": {
            "prompt-b": {"quality": 0.7 + offset, "recall": 0.5},
            "prompt-a": {"quality": 0.8 + offset, "recall": 1.0},
        },
    }


def test_collect_prompt_metric_rows_aligns_methods_and_prompts() -> None:
    rows = collect_prompt_metric_rows(
        {"A0": _evaluation(), "B3": _evaluation(offset=0.1)}
    )

    assert rows == (
        {"method": "A0", "prompt_id": "prompt-a", "quality": 0.8, "recall": 1.0},
        {"method": "A0", "prompt_id": "prompt-b", "quality": 0.7, "recall": 0.5},
        {
            "method": "B3",
            "prompt_id": "prompt-a",
            "quality": pytest.approx(0.9),
            "recall": 1.0,
        },
        {
            "method": "B3",
            "prompt_id": "prompt-b",
            "quality": pytest.approx(0.8),
            "recall": 0.5,
        },
    )


def test_collect_prompt_metric_rows_rejects_unpaired_prompt_sets() -> None:
    candidate = _evaluation()
    candidate["prompt_metrics"].pop("prompt-b")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="prompt IDs"):
        collect_prompt_metric_rows({"A0": _evaluation(), "B3": candidate})


def test_collect_prompt_metric_rows_rejects_changed_metric_schema() -> None:
    candidate = _evaluation()
    candidate["prompt_metrics"]["prompt-a"].pop("recall")  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match="metric names"):
        collect_prompt_metric_rows({"A0": _evaluation(), "B3": candidate})
