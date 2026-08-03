import pytest

from novelty_distill.training.runtime import invocation_runtime


def test_invocation_runtime_records_resumed_step_span_and_rate() -> None:
    result = invocation_runtime(
        last_checkpoint="/scratch/run/checkpoint-75",
        end_step=125,
        metrics={"train_runtime": 220.0},
    )

    assert result == {
        "start_step": 75,
        "end_step": 125,
        "optimizer_steps": 50,
        "runtime_seconds": 220.0,
        "seconds_per_optimizer_step": 4.4,
    }


def test_invocation_runtime_validates_checkpoint_and_runtime_contract() -> None:
    assert (
        invocation_runtime(
            last_checkpoint=None,
            end_step=2,
            metrics={"train_runtime": 1},
        )["start_step"]
        == 0
    )

    with pytest.raises(ValueError, match="checkpoint step"):
        invocation_runtime(
            last_checkpoint="/scratch/run/checkpoint-bad",
            end_step=125,
            metrics={"train_runtime": 1.0},
        )
    finalization = invocation_runtime(
        last_checkpoint="/scratch/run/checkpoint-125",
        end_step=125,
        metrics={"train_runtime": 1.0},
    )
    assert finalization["optimizer_steps"] == 0
    assert finalization["seconds_per_optimizer_step"] is None
    with pytest.raises(ValueError, match="cannot end before"):
        invocation_runtime(
            last_checkpoint="/scratch/run/checkpoint-126",
            end_step=125,
            metrics={"train_runtime": 1.0},
        )
    with pytest.raises(ValueError, match="positive train_runtime"):
        invocation_runtime(last_checkpoint=None, end_step=1, metrics={"train_runtime": 0})
