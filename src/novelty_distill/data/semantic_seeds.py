"""Deterministic input-level randomness for coherent trajectory selection."""

import hashlib
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class GaussianSeedSpec(BaseModel):
    """Frozen controls for a quantized Gaussian input seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimensions: int = Field(gt=0, le=32)
    bins: int = Field(ge=2, le=9)
    scale: float = Field(gt=0, le=4)
    salt: str = Field(min_length=1)


def gaussian_seed_values(
    *,
    prompt_id: str,
    sample_index: int,
    generation_seed: int,
    spec: GaussianSeedSpec,
) -> tuple[int, ...]:
    """Draw a reproducible Gaussian vector and quantize it into balanced bins."""

    if not prompt_id.strip():
        raise ValueError("Gaussian seed requires a non-empty prompt ID")
    if sample_index < 0:
        raise ValueError("Gaussian seed sample index must be nonnegative")
    if generation_seed < 0:
        raise ValueError("Gaussian generation seed must be nonnegative")
    encoded = f"{spec.salt}\0{prompt_id}\0{sample_index}\0{generation_seed}".encode()
    rng_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")
    generator = np.random.Generator(np.random.PCG64(rng_seed))
    values = generator.normal(0.0, spec.scale, spec.dimensions)
    normal = NormalDist()
    cut_points = np.asarray(
        [normal.inv_cdf(index / spec.bins) for index in range(1, spec.bins)],
        dtype=np.float64,
    )
    quantized = np.digitize(values, cut_points) - spec.bins // 2
    return tuple(int(value) for value in quantized)


def render_seed_prefix(values: Sequence[int]) -> str:
    """Serialize a seed without adding semantic information about the task."""

    if not values:
        raise ValueError("seed prefix requires at least one value")
    encoded = "-".join(
        "Z0" if value == 0 else f"{'P' if value > 0 else 'N'}{abs(value)}"
        for value in values
    )
    return (
        f"Exploration seed: {encoded}\n"
        "Use this arbitrary seed only to choose one coherent approach; do not mention it."
    )


def condition_prompt(prompt: str, *, values: Sequence[int]) -> str:
    """Prepend one trajectory seed to an otherwise unchanged prompt."""

    if not prompt.strip():
        raise ValueError("conditioned prompt must be non-empty")
    return f"{render_seed_prefix(values)}\n\n{prompt}"
