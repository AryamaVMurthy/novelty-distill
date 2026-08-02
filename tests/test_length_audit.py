import pytest

from novelty_distill.training.length_audit import summarize_token_lengths


def test_length_summary_exposes_context_overflow_and_tail() -> None:
    summary = summarize_token_lengths((10, 20, 30, 40, 200), max_length=50)

    assert summary == {
        "count": 5,
        "mean": 60.0,
        "median": 30.0,
        "p95": pytest.approx(168.0),
        "max": 200,
        "over_limit_count": 1,
        "over_limit_rate": 0.2,
        "tokens_over_limit": 150,
    }


def test_length_summary_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize_token_lengths((), max_length=10)
    with pytest.raises(ValueError, match="non-negative"):
        summarize_token_lengths((1, -1), max_length=10)
    with pytest.raises(ValueError, match="positive"):
        summarize_token_lengths((1,), max_length=0)
