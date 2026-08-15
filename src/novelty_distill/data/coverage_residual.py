"""Select valid teacher trajectories missing from a student's semantic support."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CoverageCandidate(BaseModel):
    """One teacher trajectory with its frozen quality and semantic embedding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    quality_score: float = Field(ge=0, le=1)
    embedding: tuple[float, ...] = Field(min_length=1)


class StudentProbe(BaseModel):
    """One untouched-student semantic probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    embedding: tuple[float, ...] = Field(min_length=1)


class CoverageResidualParameters(BaseModel):
    """Frozen vMF-kernel and quality-gate controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kappa: float = Field(gt=0)
    gamma: float = Field(ge=0)
    epsilon: float = Field(gt=0)
    minimum_quality: float = Field(ge=0, le=1)


class ResidualSelection(BaseModel):
    """Selected candidate and its normalized residual statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: CoverageCandidate
    density: float = Field(ge=0)
    weight: float = Field(ge=0, le=1)


def _normalized_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} embeddings must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} embeddings must be finite")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} embeddings must be normalized")
    return matrix


def vmf_similarity_density(
    teacher_embeddings: np.ndarray,
    student_embeddings: np.ndarray,
    *,
    kappa: float,
) -> np.ndarray:
    """Estimate student mass at teacher points with a cosine/vMF kernel."""

    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("vMF kappa must be positive and finite")
    teacher = _normalized_matrix(teacher_embeddings, name="teacher")
    student = _normalized_matrix(student_embeddings, name="student")
    if teacher.shape[1] != student.shape[1]:
        raise ValueError("teacher and student embedding dimensions must match")
    similarities = np.clip(teacher @ student.T, -1.0, 1.0)
    return np.exp(kappa * (similarities - 1.0)).mean(axis=1)


def residual_weights(
    quality: np.ndarray,
    density: np.ndarray,
    *,
    minimum_quality: float,
    gamma: float,
    epsilon: float,
) -> np.ndarray:
    """Normalize quality-gated inverse-student-density target weights."""

    quality_values = np.asarray(quality, dtype=np.float64)
    density_values = np.asarray(density, dtype=np.float64)
    if quality_values.ndim != 1 or quality_values.shape != density_values.shape:
        raise ValueError("quality and density must be equal-length vectors")
    if not quality_values.size:
        raise ValueError("residual weighting requires teacher candidates")
    if not np.isfinite(quality_values).all() or np.any((quality_values < 0) | (quality_values > 1)):
        raise ValueError("quality values must be finite and between zero and one")
    if not np.isfinite(density_values).all() or np.any(density_values < 0):
        raise ValueError("density values must be finite and nonnegative")
    if not 0 <= minimum_quality <= 1:
        raise ValueError("minimum quality must be between zero and one")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("residual gamma must be finite and nonnegative")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("residual epsilon must be positive and finite")
    valid = quality_values >= minimum_quality
    if not valid.any():
        raise ValueError("no valid teacher candidate passes the quality gate")
    unnormalized = np.where(
        valid,
        quality_values * np.power(density_values + epsilon, -gamma),
        0.0,
    )
    total = float(unnormalized.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("residual weights cannot be normalized")
    return unnormalized / total


def select_residual_target(
    candidates: Sequence[CoverageCandidate],
    probes: Sequence[StudentProbe],
    *,
    parameters: CoverageResidualParameters,
) -> ResidualSelection:
    """Select the highest-weight valid missing trajectory for one prompt."""

    if not candidates or not probes:
        raise ValueError("residual selection requires teacher candidates and student probes")
    prompt_ids = {candidate.prompt_id for candidate in candidates}
    if len(prompt_ids) != 1 or {probe.prompt_id for probe in probes} != prompt_ids:
        raise ValueError("residual selection inputs must share exactly one prompt ID")
    sample_indices = [candidate.sample_index for candidate in candidates]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("teacher candidate sample indices must be unique")
    teacher = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float64)
    student = np.asarray([probe.embedding for probe in probes], dtype=np.float64)
    density = vmf_similarity_density(teacher, student, kappa=parameters.kappa)
    weights = residual_weights(
        np.asarray([candidate.quality_score for candidate in candidates]),
        density,
        minimum_quality=parameters.minimum_quality,
        gamma=parameters.gamma,
        epsilon=parameters.epsilon,
    )
    selected_index = max(
        range(len(candidates)),
        key=lambda index: (float(weights[index]), -candidates[index].sample_index),
    )
    return ResidualSelection(
        candidate=candidates[selected_index],
        density=float(density[selected_index]),
        weight=float(weights[selected_index]),
    )


def build_coverage_residual_artifact(
    candidates: Sequence[CoverageCandidate],
    probes: Sequence[StudentProbe],
    *,
    parameters: CoverageResidualParameters,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Build a versioned best-one target artifact and per-prompt audit."""

    if not source_hashes or any(
        not key or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for key, value in source_hashes.items()
    ):
        raise ValueError("source hashes must be named lowercase SHA-256 digests")
    grouped_candidates: defaultdict[str, list[CoverageCandidate]] = defaultdict(list)
    grouped_probes: defaultdict[str, list[StudentProbe]] = defaultdict(list)
    for candidate in candidates:
        grouped_candidates[candidate.prompt_id].append(candidate)
    for probe in probes:
        grouped_probes[probe.prompt_id].append(probe)
    if not grouped_candidates or set(grouped_candidates) != set(grouped_probes):
        raise ValueError("teacher and student probe prompt sets must match and be non-empty")

    targets: dict[str, dict[str, list[str]]] = {}
    audit: dict[str, dict[str, object]] = {}
    for prompt_id in sorted(grouped_candidates):
        ordered = tuple(
            sorted(grouped_candidates[prompt_id], key=lambda candidate: candidate.sample_index)
        )
        selection = select_residual_target(
            ordered,
            tuple(grouped_probes[prompt_id]),
            parameters=parameters,
        )
        targets[prompt_id] = {
            "best1": [selection.candidate.text],
            "all8": [candidate.text for candidate in ordered],
        }
        audit[prompt_id] = {
            "selected_sample_index": selection.candidate.sample_index,
            "selected_quality": selection.candidate.quality_score,
            "selected_density": selection.density,
            "selected_weight": selection.weight,
        }
    return {
        "schema_version": 1,
        "targets": targets,
        "provenance": {
            "selector": parameters.model_dump(mode="json"),
            "source_hashes": dict(sorted(source_hashes.items())),
            "num_prompts": len(targets),
            "num_teacher_candidates": len(candidates),
            "num_student_probes": len(probes),
        },
        "audit": audit,
    }
