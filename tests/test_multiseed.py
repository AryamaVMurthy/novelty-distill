import pytest

from novelty_distill.evaluation.multiseed import analyze_checkpoint_seed_contrasts


def _evaluation(values: tuple[float, float]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "prompt_metrics": {
            "p1": {"score": values[0]},
            "p2": {"score": values[1]},
        },
        "prompt_metrics_by_threshold": {
            "0.900": {
                "p1": {"recall": 0.5},
                "p2": {"recall": 1.0},
            }
        },
    }


def test_multiseed_contrasts_use_checkpoint_seed_as_replication_unit() -> None:
    result = analyze_checkpoint_seed_contrasts(
        evaluations={
            "B2a": {
                17: _evaluation((2.0, 3.0)),
                29: _evaluation((3.0, 4.0)),
                43: _evaluation((4.0, 5.0)),
            },
            "B2b": {
                17: _evaluation((3.0, 4.0)),
                29: _evaluation((5.0, 6.0)),
                43: _evaluation((4.0, 5.0)),
            },
        },
        controls={"A0": _evaluation((1.0, 2.0))},
        seeds=(17, 29, 43),
        contrasts=(
            {"id": "B2a-vs-A0", "reference": "A0", "treatment": "B2a"},
            {"id": "B2b-vs-B2a", "reference": "B2a", "treatment": "B2b"},
        ),
        metrics=("score",),
        bootstrap_samples=100,
        seed=7,
    )

    first = result["results"]["score"]["B2a-vs-A0"]
    assert first["checkpoint_seed_n"] == 3
    assert first["mean_difference"] == pytest.approx(2.0)
    assert first["seed_standard_deviation"] == pytest.approx(1.0)
    assert [row["mean_difference"] for row in first["per_seed"]] == [1.0, 2.0, 3.0]
    second = result["results"]["score"]["B2b-vs-B2a"]
    assert second["mean_difference"] == pytest.approx(1.0)
    assert second["sign_consistency"] == {"negative": 0, "zero": 1, "positive": 2}


def test_multiseed_contrasts_reject_incomplete_seed_matrix() -> None:
    with pytest.raises(ValueError, match="exactly the expected seeds"):
        analyze_checkpoint_seed_contrasts(
            evaluations={"B2a": {17: _evaluation((2.0, 3.0))}},
            controls={"A0": _evaluation((1.0, 2.0))},
            seeds=(17, 29, 43),
            contrasts=(
                {"id": "B2a-vs-A0", "reference": "A0", "treatment": "B2a"},
            ),
            metrics=("score",),
            bootstrap_samples=100,
            seed=7,
        )
