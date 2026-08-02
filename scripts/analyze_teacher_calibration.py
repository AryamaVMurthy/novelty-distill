#!/usr/bin/env python3
"""Aggregate prompt-paired teacher calibration outputs and uncertainty."""

import argparse
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from novelty_distill.data.teacher_views import TeacherGeneration
from novelty_distill.evaluation.calibration_analysis import summarize_calibration_prompt
from novelty_distill.evaluation.statistics import holm_adjust, paired_bootstrap
from novelty_distill.generation.sglang import GenerationSpec, load_prompt_shard

SCALAR_METRICS = (
    "quality_mean",
    "quality_standard_deviation",
    "semantic_clusters",
    "quality_adjusted_coverage",
    "length_termination_rate",
    "completion_tokens_mean",
    "unique_text_rate",
)
COMPARISON_METRICS = (
    "quality_mean",
    "semantic_clusters",
    "quality_adjusted_coverage",
    "length_termination_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--reference")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _finite_estimate(estimate: Any) -> dict[str, Any]:
    payload = asdict(estimate)
    if not math.isfinite(payload["effect_size"]):
        payload["effect_size"] = None
    return payload


def _load_condition(
    *, condition: str, generation_root: Path, cluster_root: Path
) -> dict[str, dict[str, Any]]:
    condition_dir = generation_root / condition
    spec = GenerationSpec.model_validate_json(
        (condition_dir / "generation-config.json").read_text(encoding="utf-8")
    )
    generated: dict[str, tuple[Any, ...]] = {}
    for shard in sorted((condition_dir / "shards").glob("*.json")):
        records = load_prompt_shard(shard, spec)
        generated[records[0].prompt_id] = records

    cluster_path = cluster_root / f"{condition}.jsonl"
    clustered: dict[str, list[TeacherGeneration]] = {}
    with cluster_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = TeacherGeneration.model_validate_json(line)
                clustered.setdefault(record.prompt_id, []).append(record)
    metadata = json.loads(
        cluster_path.with_suffix(cluster_path.suffix + ".metadata.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = metadata["prompt_diagnostics"]
    if generated.keys() != clustered.keys() or generated.keys() != diagnostics.keys():
        raise ValueError(f"prompt sets do not align for condition {condition}")
    return {
        prompt_id: summarize_calibration_prompt(
            generation_records=generated[prompt_id],
            clustered_records=clustered[prompt_id],
            diagnostics=diagnostics[prompt_id],
        )
        for prompt_id in sorted(generated)
    }


def _mean_thresholds(
    prompt_summaries: dict[str, dict[str, Any]], field: str
) -> dict[str, float]:
    thresholds = next(iter(prompt_summaries.values()))[field].keys()
    return {
        threshold: statistics.fmean(
            summary[field][threshold] for summary in prompt_summaries.values()
        )
        for threshold in thresholds
    }


def _mean_judge_dimensions(
    prompt_summaries: dict[str, dict[str, Any]],
) -> dict[str, float]:
    dimension_names = next(iter(prompt_summaries.values()))[
        "judge_dimension_means"
    ].keys()
    return {
        name: statistics.fmean(
            summary["judge_dimension_means"][name]
            for summary in prompt_summaries.values()
        )
        for name in dimension_names
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Teacher calibration results",
        "",
        "These are calibration results, not trained-student or headline paper results.",
        "",
        "| Condition | Quality | Clusters@0.82 | QAC | Length stop | Unique text |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, result in payload["conditions"].items():
        aggregate = result["aggregate"]
        lines.append(
            f"| {condition} | {aggregate['quality_mean']:.3f} | "
            f"{aggregate['semantic_clusters']:.3f} | "
            f"{aggregate['quality_adjusted_coverage']:.3f} | "
            f"{aggregate['length_termination_rate']:.3f} | "
            f"{aggregate['unique_text_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Reference condition: `{payload['reference']}`.",
            "Paired bootstrap intervals and Holm-adjusted p-values are stored in the "
            "JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.generation_root / "study-manifest.json").read_text(encoding="utf-8")
    )
    condition_ids = tuple(manifest["conditions"])
    reference = args.reference or manifest.get("reference_condition", "t07-p08-l512")
    if reference not in condition_ids:
        raise ValueError(f"reference condition {reference!r} is absent")

    per_condition = {
        condition: _load_condition(
            condition=condition,
            generation_root=args.generation_root,
            cluster_root=args.cluster_root,
        )
        for condition in condition_ids
    }
    reference_prompts = tuple(per_condition[reference])
    if any(tuple(summaries) != reference_prompts for summaries in per_condition.values()):
        raise ValueError("calibration conditions do not share exactly the same prompt IDs")

    conditions: dict[str, Any] = {}
    for condition, prompt_summaries in per_condition.items():
        conditions[condition] = {
            "aggregate": {
                metric: statistics.fmean(
                    summary[metric] for summary in prompt_summaries.values()
                )
                for metric in SCALAR_METRICS
            }
            | {
                "raw_clusters_by_threshold": _mean_thresholds(
                    prompt_summaries, "raw_clusters_by_threshold"
                ),
                "instructed_clusters_by_threshold": _mean_thresholds(
                    prompt_summaries, "instructed_clusters_by_threshold"
                ),
                "judge_dimension_means": _mean_judge_dimensions(prompt_summaries),
            },
            "prompts": prompt_summaries,
        }

    comparisons: dict[str, Any] = {}
    for metric in COMPARISON_METRICS:
        metric_results: dict[str, Any] = {}
        p_values: dict[str, float] = {}
        reference_values = tuple(
            per_condition[reference][prompt_id][metric]
            for prompt_id in reference_prompts
        )
        for condition in condition_ids:
            if condition == reference:
                continue
            estimate = paired_bootstrap(
                reference=reference_values,
                treatment=tuple(
                    per_condition[condition][prompt_id][metric]
                    for prompt_id in reference_prompts
                ),
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            metric_results[condition] = _finite_estimate(estimate)
            p_values[condition] = estimate.p_value
        adjusted = holm_adjust(p_values)
        for condition, p_value in adjusted.items():
            metric_results[condition]["holm_p_value"] = p_value
        comparisons[metric] = metric_results

    payload = {
        "schema_version": 1,
        "study": manifest["study"],
        "reference": reference,
        "prompt_ids": list(reference_prompts),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "conditions": conditions,
        "comparisons": comparisons,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
