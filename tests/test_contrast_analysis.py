from novelty_distill.evaluation.contrasts import analyze_contrasts


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
    assert result["quality"]["C-vs-A"]["mean_difference"] < 0
    assert 0 <= result["quality"]["B-vs-A"]["holm_p_value"] <= 1
    assert result["recall"]["B-vs-A"]["n"] == 3
