"""Small, deterministic statistics helpers for paired prompt-level evaluation."""

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedEstimate:
    """Treatment-minus-reference estimate over the same prompts."""

    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    effect_size: float
    p_value: float


def paired_bootstrap(
    *,
    reference: Sequence[float],
    treatment: Sequence[float],
    samples: int = 10_000,
    seed: int = 1,
) -> PairedEstimate:
    """Estimate a paired mean difference, 95% CI, Cohen's dz, and sign-flip p-value."""

    if len(reference) != len(treatment) or not reference:
        raise ValueError("paired samples must have the same non-zero length")
    if len(reference) < 2:
        raise ValueError("paired analysis requires at least two prompts")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    values = tuple(float(value) for value in (*reference, *treatment))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired samples must contain only finite values")

    differences = tuple(
        float(candidate) - float(control)
        for control, candidate in zip(reference, treatment, strict=True)
    )
    observed = statistics.fmean(differences)
    standard_deviation = statistics.stdev(differences)
    if standard_deviation == 0:
        effect_size = math.copysign(math.inf, observed) if observed else 0.0
    else:
        effect_size = observed / standard_deviation

    rng = random.Random(seed)
    count = len(differences)
    bootstrap_means = sorted(
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    ci_low = _quantile(bootstrap_means, 0.025)
    ci_high = _quantile(bootstrap_means, 0.975)

    permutation_rng = random.Random(seed + 1)
    extreme = sum(
        abs(
            statistics.fmean(
                difference if permutation_rng.getrandbits(1) else -difference
                for difference in differences
            )
        )
        >= abs(observed)
        for _ in range(samples)
    )
    p_value = (extreme + 1) / (samples + 1)
    return PairedEstimate(
        n=count,
        mean_difference=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size=effect_size,
        p_value=p_value,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down family-wise error correction."""

    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must be finite values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    previous = 0.0
    total = len(ordered)
    for index, (label, p_value) in enumerate(ordered):
        current = min(1.0, (total - index) * p_value)
        previous = max(previous, current)
        adjusted[label] = previous
    return adjusted


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
