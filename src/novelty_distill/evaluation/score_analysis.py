"""Descriptive diagnostics for a complete set of strict score shards."""

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from novelty_distill.evaluation.teacher_annotation import JudgeSpec, QualityDimensions


def summarize_score_payloads(payloads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate judge discrimination and generation diagnostics without inferential claims."""

    materialized = tuple(payloads)
    if not materialized:
        raise ValueError("score analysis requires at least one payload")
    judge = JudgeSpec.model_validate(materialized[0].get("judge")).model_dump(mode="json")
    prompt_records: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for payload in materialized:
        if JudgeSpec.model_validate(payload.get("judge")).model_dump(mode="json") != judge:
            raise ValueError("score analysis payloads changed the judge specification")
        prompt_id = payload.get("prompt_id")
        records = payload.get("records")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("score analysis payload has an invalid prompt ID")
        if prompt_id in prompt_records:
            raise ValueError(f"duplicate score prompt ID {prompt_id!r}")
        if not isinstance(records, list) or not records:
            raise ValueError(f"score analysis prompt {prompt_id!r} has no records")
        prompt_records[prompt_id] = tuple(records)

    qualities: list[float] = []
    dimension_values: dict[str, list[int]] = {}
    finish_reasons: Counter[str] = Counter()
    completion_tokens: list[int] = []
    within_prompt_ranges: list[float] = []
    within_prompt_unique_scores: list[int] = []
    centered_qualities: list[float] = []
    centered_tokens: list[float] = []
    for prompt_id, records in prompt_records.items():
        prompt_qualities: list[float] = []
        for record in records:
            if record.get("prompt_id") != prompt_id:
                raise ValueError(f"score record changed prompt ID for {prompt_id!r}")
            dimensions = QualityDimensions.model_validate(record.get("dimensions"))
            raw_quality = record.get("quality_score")
            if isinstance(raw_quality, bool):
                raise ValueError("quality scores must be finite numbers in [0, 1]")
            quality = float(raw_quality)
            if not math.isfinite(quality) or not 0 <= quality <= 1:
                raise ValueError("quality scores must be finite numbers in [0, 1]")
            prompt_qualities.append(quality)
            qualities.append(quality)
            for name, value in dimensions.model_dump().items():
                dimension_values.setdefault(name, []).append(value)
            reason = record.get("finish_reason")
            tokens = record.get("completion_tokens")
            if not isinstance(reason, str) or not reason:
                raise ValueError("finish reasons must be non-empty")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                raise ValueError("completion token counts must be non-negative integers")
            finish_reasons[reason] += 1
            completion_tokens.append(tokens)
        within_prompt_ranges.append(max(prompt_qualities) - min(prompt_qualities))
        within_prompt_unique_scores.append(len(set(prompt_qualities)))
        prompt_tokens = [int(record["completion_tokens"]) for record in records]
        quality_mean = statistics.fmean(prompt_qualities)
        token_mean = statistics.fmean(prompt_tokens)
        centered_qualities.extend(value - quality_mean for value in prompt_qualities)
        centered_tokens.extend(value - token_mean for value in prompt_tokens)

    return {
        "num_prompts": len(prompt_records),
        "num_samples": len(qualities),
        "samples_per_prompt": sorted({len(records) for records in prompt_records.values()}),
        "judge": judge,
        "quality": {
            "mean": statistics.fmean(qualities),
            "median": statistics.median(qualities),
            "population_stdev": statistics.pstdev(qualities),
            "min": min(qualities),
            "max": max(qualities),
            "ceiling_rate": sum(value == 1 for value in qualities) / len(qualities),
            "score_counts": {
                f"{value:.12g}": count for value, count in sorted(Counter(qualities).items())
            },
            "within_prompt_range_mean": statistics.fmean(within_prompt_ranges),
            "within_prompt_unique_scores_mean": statistics.fmean(
                within_prompt_unique_scores
            ),
        },
        "dimensions": {
            name: {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "ceiling_rate": sum(value == 5 for value in values) / len(values),
                "rating_counts": {
                    str(value): count for value, count in sorted(Counter(values).items())
                },
            }
            for name, values in sorted(dimension_values.items())
        },
        "generation": {
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "length_stop_rate": finish_reasons["length"] / len(completion_tokens),
            "completion_tokens_mean": statistics.fmean(completion_tokens),
            "completion_tokens_median": statistics.median(completion_tokens),
            "completion_tokens_max": max(completion_tokens),
            "quality_length_pearson": _pearson(qualities, completion_tokens),
            "within_prompt_quality_length_pearson": _pearson(
                centered_qualities, centered_tokens
            ),
        },
    }


def render_score_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Render deterministic descriptive findings with the claim boundary inline."""

    quality = summary["quality"]
    dimensions = summary["dimensions"]
    generation = summary["generation"]
    lines = [
        "# Scored generation diagnostics",
        "",
        (
            f"This run contains {int(summary['num_prompts']):,} prompts and "
            f"{int(summary['num_samples']):,} samples. The fixed rubric mean is a reproducible "
            "quality proxy, not a novelty score or expert scientific judgment."
        ),
        "",
        "## Quality-score discrimination",
        "",
        "| Mean | Median | Population SD | Min | Max | Ceiling rate | Mean within-prompt range |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {_number(quality['mean'])} | {_number(quality['median'])} | "
            f"{_number(quality['population_stdev'])} | {_number(quality['min'])} | "
            f"{_number(quality['max'])} | {_number(quality['ceiling_rate'])} | "
            f"{_number(quality['within_prompt_range_mean'])} |"
        ),
        "",
        "## Rubric dimensions",
        "",
        "| Dimension | Mean | Median | Rating-5 rate | Rating counts |",
        "|---|---:|---:|---:|---|",
    ]
    if git_commit := summary.get("git_commit"):
        lines[2:2] = [f"Producer Git commit: `{git_commit}`.", ""]
    for name, values in dimensions.items():
        counts = ", ".join(f"{rating}:{count}" for rating, count in values["rating_counts"].items())
        lines.append(
            f"| `{name}` | {_number(values['mean'])} | {_number(values['median'])} | "
            f"{_number(values['ceiling_rate'])} | {counts} |"
        )
    finish_counts = ", ".join(
        f"{reason}:{count}" for reason, count in generation["finish_reason_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Generation diagnostics",
            "",
            f"- Finish reasons: {finish_counts}.",
            f"- Length-stop rate: {_number(generation['length_stop_rate'])}.",
            (
                "- Completion tokens (mean / median / max): "
                f"{_number(generation['completion_tokens_mean'])} / "
                f"{_number(generation['completion_tokens_median'])} / "
                f"{int(generation['completion_tokens_max'])}."
            ),
            (
                "- Quality–length Pearson correlation (global / prompt-centered): "
                f"{_optional_number(generation['quality_length_pearson'])} / "
                f"{_optional_number(generation['within_prompt_quality_length_pearson'])}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: Any) -> str:
    return f"{float(value):.6g}"


def _optional_number(value: Any) -> str:
    return "NA" if value is None else _number(value)


def _pearson(left: list[float], right: list[float | int]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Pearson inputs must have the same non-zero length")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return sum(
        a * b for a, b in zip(left_centered, right_centered, strict=True)
    ) / denominator
