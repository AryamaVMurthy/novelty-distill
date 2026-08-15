"""Diagnostics for controlled stochastic planning and future commitment."""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def summarize_roll_control(
    *,
    roll_ids: Iterable[str],
    semantic_modes: Iterable[str],
    valid: Iterable[bool],
) -> dict[str, Any]:
    """Separate useful between-roll diversity from within-roll decode noise.

    Mutual information is high only when changing the explicit input roll changes
    semantic mode predictably. Ordinary sampling noise increases conditional
    entropy instead.
    """

    rolls = tuple(str(value) for value in roll_ids)
    modes = tuple(str(value) for value in semantic_modes)
    validity = tuple(bool(value) for value in valid)
    if not rolls or len({len(rolls), len(modes), len(validity)}) != 1:
        raise ValueError(
            "roll IDs, semantic modes, and validity must have the same non-zero length"
        )
    if any(not value for value in (*rolls, *modes)):
        raise ValueError("roll IDs and semantic modes must be non-empty")

    grouped: dict[str, list[str]] = defaultdict(list)
    for roll, mode in zip(rolls, modes, strict=True):
        grouped[roll].append(mode)

    overall_entropy = _entropy(modes)
    conditional_entropy = sum(
        len(group) / len(modes) * _entropy(group) for group in grouped.values()
    )
    pair_agreements: list[float] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        counts = Counter(group)
        agreeing_pairs = sum(count * (count - 1) // 2 for count in counts.values())
        total_pairs = len(group) * (len(group) - 1) // 2
        pair_agreements.append(agreeing_pairs / total_pairs)
    if not pair_agreements:
        raise ValueError("roll-control diagnostics require at least two replicates per roll")

    return {
        "samples": len(modes),
        "rolls": len(grouped),
        "validity_rate": sum(validity) / len(validity),
        "mode_entropy_bits": overall_entropy,
        "conditional_mode_entropy_bits": conditional_entropy,
        "roll_mode_mutual_information_bits": max(
            0.0, overall_entropy - conditional_entropy
        ),
        "within_roll_mode_agreement": sum(pair_agreements) / len(pair_agreements),
    }


def future_commitment_curve(
    *,
    checkpoint_predictions: Mapping[int, Sequence[str]],
    realized_modes: Sequence[str],
) -> dict[int, float]:
    """Measure how early prefix rollouts predict each completion's realized mode."""

    realized = tuple(str(value) for value in realized_modes)
    if not realized:
        raise ValueError("realized modes must be non-empty")
    result: dict[int, float] = {}
    for checkpoint, predictions in sorted(checkpoint_predictions.items()):
        predicted = tuple(str(value) for value in predictions)
        if checkpoint < 0:
            raise ValueError("checkpoint positions must be non-negative")
        if len(predicted) != len(realized):
            raise ValueError("checkpoint predictions must align with realized modes")
        result[int(checkpoint)] = sum(
            left == right for left, right in zip(predicted, realized, strict=True)
        ) / len(realized)
    if not result:
        raise ValueError("at least one checkpoint is required")
    return result


def _entropy(values: Sequence[str]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())
