"""Small, deterministic statistics helpers for paired prompt-level evaluation."""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


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

    if len(reference) != len(treatment) or len(reference) == 0:
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

    count = len(differences)
    difference_array = np.asarray(differences, dtype=np.float64)
    chunk_size = min(1_000, samples)
    bootstrap_rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    permutation_rng = np.random.default_rng(seed + 1)
    extreme = 0
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        chunk = stop - start
        indices = bootstrap_rng.integers(0, count, size=(chunk, count))
        bootstrap_means[start:stop] = difference_array[indices].mean(axis=1)
        signs = permutation_rng.integers(0, 2, size=(chunk, count), dtype=np.int8) * 2 - 1
        permuted_means = (signs * difference_array).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted_means) >= abs(observed)))
    ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975), method="linear")
    p_value = (extreme + 1) / (samples + 1)
    return PairedEstimate(
        n=count,
        mean_difference=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
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
