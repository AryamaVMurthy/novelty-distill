import pytest

from novelty_distill.evaluation.statistics import holm_adjust, paired_bootstrap


def test_paired_bootstrap_reports_prompt_level_difference_and_interval() -> None:
    estimate = paired_bootstrap(
        reference=(0.1, 0.2, 0.3, 0.4),
        treatment=(0.2, 0.4, 0.6, 0.8),
        samples=2_000,
        seed=7,
    )

    assert estimate.mean_difference == pytest.approx(0.25)
    assert estimate.effect_size > 1
    assert estimate.ci_low > 0
    assert estimate.ci_high >= estimate.ci_low
    assert 0 <= estimate.p_value <= 1


def test_paired_bootstrap_rejects_unpaired_or_degenerate_values() -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        paired_bootstrap(reference=(1.0,), treatment=(), samples=100, seed=1)
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap(
            reference=(1.0, 2.0), treatment=(float("nan"), 3.0), samples=100, seed=1
        )
    with pytest.raises(ValueError, match="at least two"):
        paired_bootstrap(reference=(1.0,), treatment=(2.0,), samples=100, seed=1)


def test_holm_adjust_is_monotone_and_preserves_labels() -> None:
    adjusted = holm_adjust({"method-b": 0.04, "method-a": 0.01, "method-c": 0.03})

    assert set(adjusted) == {"method-a", "method-b", "method-c"}
    assert adjusted["method-a"] == pytest.approx(0.03)
    assert adjusted["method-c"] == pytest.approx(0.06)
    assert adjusted["method-b"] == pytest.approx(0.06)


def test_holm_adjust_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust({"bad": 1.1})
