"""Deterministic views derived from one permanent teacher-generation set."""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TeacherGeneration(BaseModel):
    """One scored and semantically clustered teacher sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    quality_score: float = Field(ge=0, le=1)
    cluster_id: str = Field(min_length=1)


TeacherView = Literal["random1", "best1", "mode1", "diverse4", "all8"]


def build_teacher_target_artifact(
    generations: Iterable[TeacherGeneration], *, seed: int
) -> dict[str, object]:
    """Build the single versioned target artifact consumed by all teacher baselines."""

    grouped: defaultdict[str, list[TeacherGeneration]] = defaultdict(list)
    for generation in generations:
        grouped[generation.prompt_id].append(generation)
    if not grouped:
        raise ValueError("teacher target artifact requires generations")

    targets: dict[str, dict[str, list[str]]] = {}
    for prompt_id in sorted(grouped):
        prompt_generations = tuple(grouped[prompt_id])
        if len(prompt_generations) != 8:
            raise ValueError(f"prompt {prompt_id} requires exactly eight permanent samples")
        if sorted(generation.sample_index for generation in prompt_generations) != list(range(8)):
            raise ValueError(f"prompt {prompt_id} requires sample indices 0 through 7")
        targets[prompt_id] = {
            view: [
                generation.text
                for generation in derive_teacher_view(
                    prompt_generations,
                    view=view,
                    seed=seed,
                )
            ]
            for view in ("random1", "best1", "mode1", "diverse4", "all8")
        }
    return {"schema_version": 1, "targets": targets}


def validate_teacher_target_artifact(
    artifact: Mapping[str, object],
    *,
    generations: Iterable[TeacherGeneration],
    seed: int,
    expected_prompt_ids: set[str] | None = None,
) -> dict[str, int]:
    """Prove that every stored view reconstructs from one frozen generation set."""

    if artifact.get("schema_version") != 1:
        raise ValueError("teacher-target artifact has an unsupported schema")
    generation_tuple = tuple(generations)
    expected = build_teacher_target_artifact(generation_tuple, seed=seed)
    if artifact.get("targets") != expected["targets"]:
        raise ValueError("teacher-target views do not reconstruct from frozen generations")
    target_mapping = expected["targets"]
    assert isinstance(target_mapping, dict)
    actual_prompt_ids = set(target_mapping)
    if expected_prompt_ids is not None and actual_prompt_ids != expected_prompt_ids:
        missing = sorted(expected_prompt_ids - actual_prompt_ids)
        extra = sorted(actual_prompt_ids - expected_prompt_ids)
        raise ValueError(
            "teacher-target prompt IDs differ from expected prompts: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return {
        "num_prompts": len(actual_prompt_ids),
        "num_generations": len(generation_tuple),
        "seed": seed,
    }


def derive_teacher_view(
    generations: Iterable[TeacherGeneration], *, view: TeacherView, seed: int
) -> tuple[TeacherGeneration, ...]:
    """Derive a training view without regenerating or changing teacher samples."""

    ordered = tuple(sorted(generations, key=lambda generation: generation.sample_index))
    if not ordered:
        raise ValueError("teacher view requires at least one generation")
    if len({generation.prompt_id for generation in ordered}) != 1:
        raise ValueError("teacher view must contain exactly one prompt id")
    indices = [generation.sample_index for generation in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError("teacher sample indices must be unique")

    quality_ranked = tuple(
        sorted(ordered, key=lambda generation: (-generation.quality_score, generation.sample_index))
    )
    if view == "all8":
        return ordered
    if view == "best1":
        return quality_ranked[:1]
    if view == "random1":
        digest = hashlib.sha256(f"{seed}\0{ordered[0].prompt_id}".encode()).digest()
        return (ordered[int.from_bytes(digest[:8], "big") % len(ordered)],)
    if view == "mode1":
        counts = Counter(generation.cluster_id for generation in ordered)
        best_by_cluster = {
            cluster_id: next(
                generation
                for generation in quality_ranked
                if generation.cluster_id == cluster_id
            )
            for cluster_id in counts
        }
        selected_cluster = min(
            counts,
            key=lambda cluster_id: (
                -counts[cluster_id],
                -best_by_cluster[cluster_id].quality_score,
                cluster_id,
            ),
        )
        return (best_by_cluster[selected_cluster],)
    if view == "diverse4":
        if len(ordered) < 4:
            raise ValueError("diverse4 requires at least four teacher generations")
        selected: list[TeacherGeneration] = []
        seen_clusters: set[str] = set()
        for generation in quality_ranked:
            if generation.cluster_id not in seen_clusters:
                selected.append(generation)
                seen_clusters.add(generation.cluster_id)
            if len(selected) == 4:
                return tuple(selected)
        for generation in quality_ranked:
            if generation not in selected:
                selected.append(generation)
            if len(selected) == 4:
                return tuple(selected)
        raise AssertionError("unreachable teacher-view selection state")
    raise ValueError(f"unsupported teacher view {view}")
