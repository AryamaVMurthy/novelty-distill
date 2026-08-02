"""Pure summaries for pre-run tokenizer context audits."""

import statistics
from collections.abc import Iterable

from novelty_distill.data.tomato import CanonicalExample


def select_canonical_json_lines(
    lines: Iterable[str], *, ids: Iterable[str]
) -> tuple[str, ...]:
    """Select exact canonical JSON records in an explicit ID order."""

    requested = tuple(str(value).strip() for value in ids)
    if not requested or any(not value for value in requested):
        raise ValueError("requested IDs must be non-empty")
    if len(requested) != len(set(requested)):
        raise ValueError("requested IDs must be unique")
    by_id: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        example = CanonicalExample.model_validate_json(stripped)
        if example.id in by_id:
            raise ValueError(f"duplicate canonical ID {example.id}")
        by_id[example.id] = stripped
    missing = tuple(identifier for identifier in requested if identifier not in by_id)
    if missing:
        raise ValueError(f"missing requested canonical IDs: {missing}")
    return tuple(by_id[identifier] for identifier in requested)


def summarize_token_lengths(
    lengths: Iterable[int], *, max_length: int
) -> dict[str, float | int]:
    """Summarize exact token lengths and overflow at one declared context limit."""

    values = tuple(int(value) for value in lengths)
    if not values:
        raise ValueError("token lengths must be non-empty")
    if any(value < 0 for value in values):
        raise ValueError("token lengths must be non-negative")
    if max_length <= 0:
        raise ValueError("max length must be positive")
    ordered = sorted(values)
    rank = 0.95 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    p95 = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
    over = tuple(value for value in values if value > max_length)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": p95,
        "max": max(values),
        "over_limit_count": len(over),
        "over_limit_rate": len(over) / len(values),
        "tokens_over_limit": sum(value - max_length for value in over),
    }
