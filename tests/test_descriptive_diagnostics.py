import pytest

from novelty_distill.evaluation.descriptive_diagnostics import (
    analyze_three_seed_descriptive_diagnostics,
)


def _evaluation(offset: float, breadth: float) -> dict[str, object]:
    prompts = {
        "p1": {
            "student_relevance_mean": 5.0,
            "student_feasibility_mean": 4.0 + offset,
            "student_soundness_mean": 4.5 + offset,
            "student_clarity_mean": 5.0,
            "student_instruction_compliance_mean": 5.0,
            "completion_tokens_mean": 100.0,
        },
        "p2": {
            "student_relevance_mean": 4.5,
            "student_feasibility_mean": 4.5 + offset,
            "student_soundness_mean": 4.0 + offset,
            "student_clarity_mean": 5.0,
            "student_instruction_compliance_mean": 5.0,
            "completion_tokens_mean": 200.0,
        },
    }
    return {
        "schema_version": 1,
        "num_prompts": 2,
        "prompt_metrics": prompts,
        "overall": {"length_stop_rate": 0.0},
        "threshold_sensitivity": {
            "0.900": {
                "quality_qualified_semantic_yield": breadth,
                "cluster_jsd": 1.0 - breadth / 10,
            },
            "0.940": {
                "quality_qualified_semantic_yield": breadth + 1,
                "cluster_jsd": 0.9 - breadth / 10,
            },
        },
    }


def test_descriptive_diagnostics_separate_saturation_from_threshold_stability() -> None:
    result = analyze_three_seed_descriptive_diagnostics(
        evaluations={
            "C1": {
                17: _evaluation(0.0, 2.0),
                29: _evaluation(0.1, 2.2),
                43: _evaluation(-0.1, 1.8),
            },
            "C2": {
                17: _evaluation(0.0, 1.5),
                29: _evaluation(0.1, 1.7),
                43: _evaluation(-0.1, 1.3),
            },
        },
        controls={"A0": _evaluation(0.0, 1.0)},
        seeds=(17, 29, 43),
        contrasts=({"id": "C2-vs-C1", "reference": "C1", "treatment": "C2"},),
        semantic_metrics={
            "quality_qualified_semantic_yield": "higher",
            "cluster_jsd": "lower",
        },
    )

    clarity = result["judge_prompt_mean_diagnostics"]["trained_methods"]["C1"]["across_seeds"][
        "dimensions"
    ]["student_clarity_mean"]
    assert clarity["prompt_ceiling_rate"] == 1.0
    feasibility = result["judge_prompt_mean_diagnostics"]["trained_methods"]["C1"]["across_seeds"][
        "dimensions"
    ]["student_feasibility_mean"]
    assert feasibility["mean"] == pytest.approx(4.25)
    assert feasibility["mean_seed_sd"] == pytest.approx(0.1)

    breadth = result["semantic_contrast_stability"]["C2-vs-C1"]["quality_qualified_semantic_yield"]
    assert breadth["stable_direction"] == "unfavorable"
    assert breadth["unfavorable_threshold_count"] == 2
    assert breadth["thresholds"]["0.940"]["mean_difference"] == pytest.approx(-0.5)

    jsd = result["semantic_contrast_stability"]["C2-vs-C1"]["cluster_jsd"]
    assert jsd["stable_direction"] == "unfavorable"


def test_descriptive_diagnostics_reject_misaligned_prompts() -> None:
    treatment = _evaluation(0.0, 2.0)
    treatment["prompt_metrics"]["p3"] = treatment["prompt_metrics"].pop("p2")

    with pytest.raises(ValueError, match="prompt ids are not aligned"):
        analyze_three_seed_descriptive_diagnostics(
            evaluations={"C1": {17: treatment}},
            controls={"A0": _evaluation(0.0, 1.0)},
            seeds=(17,),
            contrasts=({"id": "C1-vs-A0", "reference": "A0", "treatment": "C1"},),
            semantic_metrics={"quality_qualified_semantic_yield": "higher"},
        )
