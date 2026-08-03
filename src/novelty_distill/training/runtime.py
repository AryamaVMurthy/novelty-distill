"""Comparable runtime provenance for preemptible trainer invocations."""

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def invocation_runtime(
    *,
    last_checkpoint: str | None,
    end_step: int,
    metrics: Mapping[str, Any],
) -> dict[str, float | int]:
    """Describe only the optimizer-step span covered by one trainer invocation."""

    start_step = 0
    if last_checkpoint is not None:
        match = re.fullmatch(r"checkpoint-([0-9]+)", Path(last_checkpoint).name)
        if match is None:
            raise ValueError("last checkpoint has no numeric checkpoint step")
        start_step = int(match.group(1))
    if isinstance(end_step, bool) or not isinstance(end_step, int) or end_step <= start_step:
        raise ValueError("trainer invocation must end after its start step")
    runtime = metrics.get("train_runtime")
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, int | float)
        or not math.isfinite(float(runtime))
        or runtime <= 0
    ):
        raise ValueError("trainer invocation requires a finite positive train_runtime")
    optimizer_steps = end_step - start_step
    runtime_seconds = float(runtime)
    return {
        "start_step": start_step,
        "end_step": end_step,
        "optimizer_steps": optimizer_steps,
        "runtime_seconds": runtime_seconds,
        "seconds_per_optimizer_step": runtime_seconds / optimizer_steps,
    }
