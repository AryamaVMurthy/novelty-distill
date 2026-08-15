"""Teacherless auxiliary inputs for prompt-conditioned whole-future prediction."""

from typing import Any


def build_teacherless_inputs(
    input_ids: Any,
    labels: Any,
    *,
    neutral_token_id: int,
) -> Any:
    """Replace visible completion tokens while retaining prompt and label positions."""

    if isinstance(neutral_token_id, bool) or not isinstance(neutral_token_id, int):
        raise ValueError("teacherless neutral token ID must be a nonnegative integer")
    if neutral_token_id < 0:
        raise ValueError("teacherless neutral token ID must be a nonnegative integer")
    if getattr(input_ids, "shape", None) != getattr(labels, "shape", None):
        raise ValueError("teacherless input and label shapes must match")
    if len(input_ids.shape) != 2:
        raise ValueError("teacherless input and labels must be rank-two batches")
    active = labels != -100
    if not bool(active.any()):
        raise ValueError("teacherless batch has no active completion labels")
    result = input_ids.clone() if hasattr(input_ids, "clone") else input_ids.copy()
    result[active] = neutral_token_id
    return result


def teacherless_cross_entropy(logits: Any, labels: Any) -> Any:
    """Compute causal CE for teacher labels from neutralized completion inputs."""

    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("teacherless logits and labels must be rank three and two")
    if logits.shape[:2] != labels.shape:
        raise ValueError("teacherless logit and label sequence shapes must match")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    active = shifted_labels != -100
    if not bool(active.any()):
        raise ValueError("teacherless batch has no shifted active completion labels")
    return functional.cross_entropy(shifted_logits[active].float(), shifted_labels[active])
