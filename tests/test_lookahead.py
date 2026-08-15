import numpy as np
import pytest

from novelty_distill.training.lookahead import (
    build_teacherless_inputs,
    combine_distillation_losses,
    resolve_neutral_token_id,
    teacherless_cross_entropy,
)


def test_teacherless_inputs_preserve_prompt_and_replace_active_completion() -> None:
    input_ids = np.asarray([[10, 11, 21, 22, 23]], dtype=np.int64)
    labels = np.asarray([[-100, -100, 21, 22, 23]], dtype=np.int64)

    teacherless = build_teacherless_inputs(input_ids, labels, neutral_token_id=7)

    assert teacherless.tolist() == [[10, 11, 7, 7, 7]]
    assert input_ids.tolist() == [[10, 11, 21, 22, 23]]
    assert labels.tolist() == [[-100, -100, 21, 22, 23]]


def test_teacherless_inputs_preserve_padding_and_truncated_tail_mask() -> None:
    input_ids = np.asarray([[0, 10, 11, 21, 22]], dtype=np.int64)
    labels = np.asarray([[-100, -100, -100, 21, -100]], dtype=np.int64)

    teacherless = build_teacherless_inputs(input_ids, labels, neutral_token_id=7)

    assert teacherless.tolist() == [[0, 10, 11, 7, 22]]


@pytest.mark.parametrize("neutral_token_id", [-1, 1.2])
def test_teacherless_inputs_reject_invalid_neutral_token(neutral_token_id: object) -> None:
    with pytest.raises(ValueError, match="neutral token"):
        build_teacherless_inputs(
            np.asarray([[10, 21]]),
            np.asarray([[-100, 21]]),
            neutral_token_id=neutral_token_id,
        )


def test_teacherless_inputs_reject_shape_mismatch_or_no_active_labels() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_teacherless_inputs(
            np.asarray([[10, 21]]),
            np.asarray([[-100]]),
            neutral_token_id=7,
        )
    with pytest.raises(ValueError, match="active completion"):
        build_teacherless_inputs(
            np.asarray([[10, 21]]),
            np.asarray([[-100, -100]]),
            neutral_token_id=7,
        )


def test_teacherless_cross_entropy_is_finite_and_uses_causal_shift() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [4.0, 0.0, 0.0],
            ]
        ]
    )
    labels = torch.tensor([[-100, 1, 2]])

    loss = teacherless_cross_entropy(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() < 0.1


def test_teacherless_cross_entropy_rejects_no_shifted_active_labels() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="active completion"):
        teacherless_cross_entropy(torch.zeros((1, 2, 3)), torch.tensor([[-100, -100]]))


def test_zero_weight_returns_the_exact_ordinary_loss_without_auxiliary() -> None:
    ordinary = object()

    assert combine_distillation_losses(ordinary, None, weight=0.0) is ordinary


def test_positive_weight_combines_ordinary_and_teacherless_losses() -> None:
    assert combine_distillation_losses(2.0, 4.0, weight=0.25) == 3.0


def test_neutral_token_must_encode_to_exactly_one_existing_token() -> None:
    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return {"!": [7], "many": [8, 9], "none": []}[text]

    assert resolve_neutral_token_id(Tokenizer(), "!") == 7
    with pytest.raises(ValueError, match="exactly one"):
        resolve_neutral_token_id(Tokenizer(), "many")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_neutral_token_id(Tokenizer(), "none")
