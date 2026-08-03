import pytest

from novelty_distill.evaluation.contrasts import (
    analyze_contrasts,
    analyze_threshold_directions,
    partition_available_contrasts,
    summarize_method_metrics,
)


def test_contrast_analysis_pairs_prompts_and_holm_adjusts_each_metric() -> None:
    rows = tuple(
        {"method": method, "prompt_id": prompt_id, "quality": value, "recall": recall}
        for method, values in {
            "A": ((0.1, 0.2), (0.2, 0.4), (0.3, 0.6)),
            "B": ((0.2, 0.4), (0.3, 0.6), (0.4, 0.8)),
            "C": ((0.1, 0.1), (0.1, 0.2), (0.2, 0.3)),
        }.items()
        for prompt_id, (value, recall) in zip(("p1", "p2", "p3"), values, strict=True)
    )

    result = analyze_contrasts(
        rows=rows,
        contrasts=(
            {"id": "B-vs-A", "reference": "A", "treatment": "B"},
            {"id": "C-vs-A", "reference": "A", "treatment": "C"},
        ),
        metrics=("quality", "recall"),
        bootstrap_samples=500,
        seed=7,
    )

    assert result["quality"]["B-vs-A"]["mean_difference"] > 0
    assert result["quality"]["B-vs-A"]["reference_mean"] == pytest.approx(0.2)
    assert result["quality"]["B-vs-A"]["treatment_mean"] == pytest.approx(0.3)
    assert result["quality"]["C-vs-A"]["mean_difference"] < 0
    assert 0 <= result["quality"]["B-vs-A"]["holm_p_value"] <= 1
    assert result["recall"]["B-vs-A"]["n"] == 3

    summaries = summarize_method_metrics(rows=rows, metrics=("quality", "recall"))
    assert summaries["A"] == {
        "n": 3,
        "quality_mean": pytest.approx(0.2),
        "quality_median": pytest.approx(0.2),
        "recall_mean": pytest.approx(0.4),
        "recall_median": pytest.approx(0.4),
    }


def test_threshold_direction_analysis_respects_metric_direction() -> None:
    rows = tuple(
        {
            "method": method,
            "threshold": threshold,
            "prompt_id": prompt_id,
            "recall": recall,
            "jsd": jsd,
        }
        for method, threshold_values in {
            "A": {
                "0.700": ((0.2, 0.4), (0.4, 0.3), (0.5, 0.2)),
                "0.820": ((0.1, 0.5), (0.3, 0.4), (0.4, 0.3)),
            },
            "B": {
                "0.700": ((0.3, 0.3), (0.4, 0.3), (0.45, 0.25)),
                "0.820": ((0.2, 0.4), (0.4, 0.3), (0.5, 0.2)),
            },
        }.items()
        for threshold, values in threshold_values.items()
        for prompt_id, (recall, jsd) in zip(("p1", "p2", "p3"), values, strict=True)
    )

    result = analyze_threshold_directions(
        rows=rows,
        contrasts=({"id": "B-vs-A", "reference": "A", "treatment": "B"},),
        metric_directions={"recall": "higher", "jsd": "lower"},
    )

    recall = result["recall"]["B-vs-A"]
    assert recall["thresholds"]["0.700"]["favorable_count"] == 1
    assert recall["thresholds"]["0.700"]["tied_count"] == 1
    assert recall["thresholds"]["0.700"]["unfavorable_count"] == 1
    assert recall["stable_mean_direction"] == "favorable"
    jsd = result["jsd"]["B-vs-A"]
    assert jsd["thresholds"]["0.700"]["favorable_count"] == 1
    assert jsd["thresholds"]["0.700"]["unfavorable_count"] == 1
    assert jsd["stable_mean_direction"] == "favorable"


def test_promoted_analysis_filters_only_absent_preregistered_contrasts() -> None:
    available, omitted = partition_available_contrasts(
        contrasts=(
            {"id": "B-vs-A", "reference": "A", "treatment": "B"},
            {"id": "C-vs-A", "reference": "A", "treatment": "C"},
        ),
        methods=("A", "B"),
    )

    assert available == (
        {"id": "B-vs-A", "reference": "A", "treatment": "B"},
    )
    assert omitted == ("C-vs-A",)

    with pytest.raises(ValueError, match="no declared contrast"):
        partition_available_contrasts(
            contrasts=(
                {"id": "C-vs-A", "reference": "A", "treatment": "C"},
            ),
            methods=("A", "B"),
        )
