import pytest

from novelty_distill.evaluation.noveltybench_matrix import paired_noveltybench_contrasts


def test_paired_noveltybench_contrasts_use_prompt_pairing() -> None:
    scores = {
        "A0": {
            "p0": {"distinct_k": 2.0, "utility_k": 3.0},
            "p1": {"distinct_k": 4.0, "utility_k": 5.0},
            "p2": {"distinct_k": 6.0, "utility_k": 7.0},
        },
        "B2b": {
            "p0": {"distinct_k": 3.0, "utility_k": 5.0},
            "p1": {"distinct_k": 6.0, "utility_k": 6.0},
            "p2": {"distinct_k": 8.0, "utility_k": 7.0},
        },
    }

    result = paired_noveltybench_contrasts(scores, baseline="A0")

    distinct = result["B2b"]["distinct_k"]
    utility = result["B2b"]["utility_k"]
    assert distinct["n_prompts"] == 3
    assert distinct["baseline_mean"] == pytest.approx(4.0)
    assert distinct["method_mean"] == pytest.approx(17 / 3)
    assert distinct["mean_delta"] == pytest.approx(5 / 3)
    assert distinct["delta_stderr"] == pytest.approx(1 / 3)
    assert utility["mean_delta"] == pytest.approx(1.0)
    assert utility["improved_prompt_fraction"] == pytest.approx(2 / 3)
    assert utility["tied_prompt_fraction"] == pytest.approx(1 / 3)


def test_paired_noveltybench_contrasts_reject_nonidentical_prompt_sets() -> None:
    scores = {
        "A0": {"p0": {"distinct_k": 2.0, "utility_k": 3.0}},
        "B2b": {"p1": {"distinct_k": 3.0, "utility_k": 4.0}},
    }

    with pytest.raises(ValueError, match="prompt IDs differ"):
        paired_noveltybench_contrasts(scores, baseline="A0")


def test_paired_noveltybench_contrasts_reject_missing_or_nonfinite_scores() -> None:
    missing = {
        "A0": {"p0": {"distinct_k": 2.0, "utility_k": 3.0}},
        "B2b": {"p0": {"distinct_k": 3.0}},
    }
    with pytest.raises(ValueError, match="required metrics"):
        paired_noveltybench_contrasts(missing, baseline="A0")

    nonfinite = {
        "A0": {"p0": {"distinct_k": 2.0, "utility_k": 3.0}},
        "B2b": {"p0": {"distinct_k": float("nan"), "utility_k": 4.0}},
    }
    with pytest.raises(ValueError, match="finite"):
        paired_noveltybench_contrasts(nonfinite, baseline="A0")
