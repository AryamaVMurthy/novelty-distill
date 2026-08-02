"""Prompt-level summaries for teacher sampling calibration."""

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from novelty_distill.data.teacher_views import TeacherGeneration
from novelty_distill.evaluation.semantic_modes import quality_adjusted_coverage
from novelty_distill.generation.sglang import GenerationRecord


def summarize_calibration_prompt(
    *,
    generation_records: Sequence[GenerationRecord],
    clustered_records: Sequence[TeacherGeneration],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate aligned sample records and calculate one prompt's calibration metrics."""

    generated = tuple(sorted(generation_records, key=lambda record: record.sample_index))
    clustered = tuple(sorted(clustered_records, key=lambda record: record.sample_index))
    generated_keys = [
        (record.prompt_id, record.sample_index, record.text) for record in generated
    ]
    clustered_keys = [
        (record.prompt_id, record.sample_index, record.text) for record in clustered
    ]
    if not generated or generated_keys != clustered_keys:
        raise ValueError("generation and clustered records must align exactly")
    prompt_id = generated[0].prompt_id
    if any(record.prompt_id != prompt_id for record in generated):
        raise ValueError("calibration prompt summary received multiple prompt ids")

    primary_embedding = diagnostics.get("primary_embedding")
    if primary_embedding != "instructed":
        raise ValueError("calibration diagnostics must declare instructed primary embeddings")
    raw = diagnostics.get("raw")
    instructed = diagnostics.get("instructed")
    if not isinstance(raw, Mapping) or not isinstance(instructed, Mapping):
        raise ValueError("calibration diagnostics require raw and instructed variants")
    raw_thresholds = raw.get("clusters_by_threshold")
    instructed_thresholds = instructed.get("clusters_by_threshold")
    if not isinstance(raw_thresholds, Mapping) or not isinstance(
        instructed_thresholds, Mapping
    ):
        raise ValueError("calibration diagnostics require threshold cluster counts")

    qualities = tuple(record.quality_score for record in clustered)
    cluster_ids = tuple(record.cluster_id for record in clustered)
    completion_tokens = tuple(
        record.completion_tokens
        for record in generated
        if record.completion_tokens is not None
    )
    return {
        "prompt_id": prompt_id,
        "samples": len(generated),
        "unique_text_rate": len({record.text for record in generated}) / len(generated),
        "length_termination_rate": sum(
            record.finish_reason == "length" for record in generated
        )
        / len(generated),
        "completion_tokens_mean": (
            statistics.fmean(completion_tokens) if completion_tokens else None
        ),
        "quality_mean": statistics.fmean(qualities),
        "quality_standard_deviation": statistics.pstdev(qualities),
        "quality_min": min(qualities),
        "quality_max": max(qualities),
        "semantic_clusters": len(set(cluster_ids)),
        "quality_adjusted_coverage": quality_adjusted_coverage(
            clusters=cluster_ids, quality_scores=qualities
        ),
        "raw_cosine_min": raw["cosine_min"],
        "raw_cosine_mean": raw["cosine_mean"],
        "raw_cosine_max": raw["cosine_max"],
        "instructed_cosine_min": instructed["cosine_min"],
        "instructed_cosine_mean": instructed["cosine_mean"],
        "instructed_cosine_max": instructed["cosine_max"],
        "raw_clusters_by_threshold": dict(raw_thresholds),
        "instructed_clusters_by_threshold": dict(instructed_thresholds),
    }
