import pytest

from novelty_distill.evaluation.forward_branches import (
    future_commitment_curve,
    summarize_roll_control,
)


def test_roll_control_rewards_across_roll_diversity_and_within_roll_stability() -> None:
    summary = summarize_roll_control(
        roll_ids=("z0", "z0", "z1", "z1", "z2", "z2", "z3", "z3"),
        semantic_modes=("a", "a", "b", "b", "c", "c", "a", "a"),
        valid=(True,) * 8,
    )

    assert summary == {
        "samples": 8,
        "rolls": 4,
        "validity_rate": 1.0,
        "mode_entropy_bits": pytest.approx(1.5),
        "conditional_mode_entropy_bits": pytest.approx(0.0),
        "roll_mode_mutual_information_bits": pytest.approx(1.5),
        "within_roll_mode_agreement": pytest.approx(1.0),
    }


def test_roll_control_detects_noise_that_is_not_controlled_by_roll() -> None:
    summary = summarize_roll_control(
        roll_ids=("z0", "z0", "z1", "z1"),
        semantic_modes=("a", "b", "a", "b"),
        valid=(True, False, True, False),
    )

    assert summary["mode_entropy_bits"] == pytest.approx(1.0)
    assert summary["conditional_mode_entropy_bits"] == pytest.approx(1.0)
    assert summary["roll_mode_mutual_information_bits"] == pytest.approx(0.0)
    assert summary["within_roll_mode_agreement"] == pytest.approx(0.0)
    assert summary["validity_rate"] == pytest.approx(0.5)


def test_future_commitment_curve_measures_early_prediction_of_realized_mode() -> None:
    curve = future_commitment_curve(
        checkpoint_predictions={
            0: ("a", "x", "c", "x"),
            16: ("a", "b", "c", "x"),
            32: ("a", "b", "c", "d"),
        },
        realized_modes=("a", "b", "c", "d"),
    )

    assert curve == {0: 0.5, 16: 0.75, 32: 1.0}


@pytest.mark.parametrize(
    "arguments",
    [
        {"roll_ids": (), "semantic_modes": (), "valid": ()},
        {"roll_ids": ("z0",), "semantic_modes": ("a", "b"), "valid": (True,)},
    ],
)
def test_roll_control_rejects_empty_or_misaligned_records(arguments: dict) -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        summarize_roll_control(**arguments)
