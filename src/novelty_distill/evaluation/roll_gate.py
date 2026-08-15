"""Validity-first comparison for ordinary and Gaussian-controlled decoding."""

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from novelty_distill.data.teacher_views import TeacherGeneration
from novelty_distill.evaluation.forward_branches import summarize_roll_control
from novelty_distill.evaluation.student_evaluation import meets_quality_qualified_gate


def compare_clustered_roll_runs(
    *, ordinary_path: Path, seeded_path: Path, input_seed_repeats: int
) -> dict[str, Any]:
    """Compare paired prompt runs and expose controlled-roll diagnostics."""

    ordinary = _load_run(ordinary_path)
    seeded = _load_run(seeded_path)
    if ordinary.keys() != seeded.keys():
        raise ValueError("ordinary and seeded runs must contain identical prompt IDs")
    ordinary_summary = _summarize(ordinary)
    seeded_summary = _summarize(seeded, repeats=input_seed_repeats)
    paired_metrics = (
        "validity_rate",
        "quality_mean",
        "semantic_clusters_mean",
        "quality_qualified_semantic_yield_mean",
        "mode_entropy_bits_mean",
    )
    return {
        "ordinary": ordinary_summary,
        "seeded": seeded_summary,
        "deltas": {
            name: seeded_summary[name] - ordinary_summary[name]
            for name in paired_metrics
        },
    }


def summarize_clustered_run(
    path: Path, *, input_seed_repeats: int | None = None
) -> dict[str, float | int]:
    """Summarize one judged, clustered generation run."""

    return _summarize(_load_run(path), repeats=input_seed_repeats)


def select_validity_first(
    summaries: Mapping[str, Mapping[str, float | int]],
    *,
    reference_id: str,
    validity_tolerance: float,
) -> dict[str, Any]:
    """Select maximum valid semantic yield inside a fixed validity margin."""

    if reference_id not in summaries:
        raise ValueError(f"missing validity reference {reference_id}")
    if not 0 <= validity_tolerance <= 1:
        raise ValueError("validity tolerance must be between zero and one")
    required = (
        "validity_rate",
        "quality_qualified_semantic_yield_mean",
        "quality_mean",
    )
    for run_id, summary in summaries.items():
        if any(name not in summary for name in required):
            raise ValueError(f"run {run_id} is missing selection metrics")
    floor = float(summaries[reference_id]["validity_rate"]) - validity_tolerance
    eligible = [
        run_id
        for run_id, summary in summaries.items()
        if float(summary["validity_rate"]) >= floor
    ]
    if not eligible:
        raise AssertionError("the reference run must satisfy its own validity floor")
    selected = min(
        eligible,
        key=lambda run_id: (
            -float(summaries[run_id]["quality_qualified_semantic_yield_mean"]),
            -float(summaries[run_id]["quality_mean"]),
            -float(summaries[run_id]["validity_rate"]),
            run_id,
        ),
    )
    return {
        "reference": reference_id,
        "validity_tolerance": validity_tolerance,
        "validity_floor": floor,
        "eligible": eligible,
        "selected": selected,
        "ranking": [
            "quality_qualified_semantic_yield_mean",
            "quality_mean",
            "validity_rate",
        ],
    }


def _load_run(path: Path) -> dict[str, tuple[tuple[TeacherGeneration, bool], ...]]:
    records = tuple(
        TeacherGeneration.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError(f"clustered run is empty: {path}")
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    diagnostics = metadata.get("prompt_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"clustered metadata has no prompt diagnostics: {metadata_path}")

    grouped: dict[str, list[TeacherGeneration]] = defaultdict(list)
    for record in records:
        grouped[record.prompt_id].append(record)
    result: dict[str, tuple[tuple[TeacherGeneration, bool], ...]] = {}
    for prompt_id, prompt_records in grouped.items():
        ordered = sorted(prompt_records, key=lambda record: record.sample_index)
        dimensions = diagnostics.get(prompt_id, {}).get("judge_dimensions")
        if not isinstance(dimensions, list) or len(dimensions) != len(ordered):
            raise ValueError(f"judge dimensions do not align for prompt {prompt_id}")
        if [record.sample_index for record in ordered] != list(range(len(ordered))):
            raise ValueError(f"sample indices are not contiguous for prompt {prompt_id}")
        result[prompt_id] = tuple(
            (record, meets_quality_qualified_gate(dimension))
            for record, dimension in zip(ordered, dimensions, strict=True)
        )
    return result


def _summarize(
    prompts: dict[str, tuple[tuple[TeacherGeneration, bool], ...]],
    *,
    repeats: int | None = None,
) -> dict[str, float | int]:
    per_prompt: list[dict[str, float]] = []
    for entries in prompts.values():
        modes = tuple(record.cluster_id for record, _ in entries)
        validity = tuple(valid for _, valid in entries)
        counts = Counter(modes)
        total = len(modes)
        item = {
            "validity_rate": sum(validity) / total,
            "quality_mean": statistics.fmean(record.quality_score for record, _ in entries),
            "semantic_clusters": float(len(counts)),
            "quality_qualified_semantic_yield": float(
                len(
                    {
                        record.cluster_id
                        for record, is_valid in entries
                        if is_valid
                    }
                )
            ),
            "mode_entropy_bits": -sum(
                (count / total) * math.log2(count / total) for count in counts.values()
            ),
        }
        if repeats is not None:
            roll = summarize_roll_control(
                roll_ids=tuple(
                    f"z{record.sample_index // repeats}" for record, _ in entries
                ),
                semantic_modes=modes,
                valid=validity,
            )
            item["roll_mode_mutual_information_bits"] = float(
                roll["roll_mode_mutual_information_bits"]
            )
            item["within_roll_mode_agreement"] = float(
                roll["within_roll_mode_agreement"]
            )
        per_prompt.append(item)

    summary: dict[str, float | int] = {
        "prompts": len(per_prompt),
        "validity_rate": statistics.fmean(item["validity_rate"] for item in per_prompt),
        "quality_mean": statistics.fmean(item["quality_mean"] for item in per_prompt),
        "semantic_clusters_mean": statistics.fmean(
            item["semantic_clusters"] for item in per_prompt
        ),
        "quality_qualified_semantic_yield_mean": statistics.fmean(
            item["quality_qualified_semantic_yield"] for item in per_prompt
        ),
        "mode_entropy_bits_mean": statistics.fmean(
            item["mode_entropy_bits"] for item in per_prompt
        ),
    }
    if repeats is not None:
        summary["roll_mode_mutual_information_bits_mean"] = statistics.fmean(
            item["roll_mode_mutual_information_bits"] for item in per_prompt
        )
        summary["within_roll_mode_agreement_mean"] = statistics.fmean(
            item["within_roll_mode_agreement"] for item in per_prompt
        )
    return summary
