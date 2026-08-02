"""Descriptive analysis of frozen teacher target views and semantic clusters."""

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from novelty_distill.data.teacher_views import (
    TeacherGeneration,
    validate_teacher_target_artifact,
)


def summarize_teacher_targets(
    *,
    generations: Iterable[TeacherGeneration],
    score_payloads: Iterable[Mapping[str, Any]],
    target_artifact: Mapping[str, object],
    cluster_metadata: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Summarize selection trade-offs without treating clusters as expert novelty labels."""

    generation_tuple = tuple(generations)
    grouped: defaultdict[str, list[TeacherGeneration]] = defaultdict(list)
    for generation in generation_tuple:
        grouped[generation.prompt_id].append(generation)
    prompt_ids = set(grouped)
    validate_teacher_target_artifact(
        target_artifact,
        generations=generation_tuple,
        seed=seed,
        expected_prompt_ids=prompt_ids,
    )
    raw_targets = target_artifact["targets"]
    if not isinstance(raw_targets, Mapping):
        raise ValueError("teacher-target artifact has no target mapping")

    score_by_text: dict[tuple[str, str], Mapping[str, Any]] = {}
    score_prompt_ids: set[str] = set()
    for payload in score_payloads:
        prompt_id = payload.get("prompt_id")
        records = payload.get("records")
        if not isinstance(prompt_id, str) or prompt_id in score_prompt_ids:
            raise ValueError("score payloads have invalid or duplicate prompt IDs")
        if not isinstance(records, list) or len(records) != 8:
            raise ValueError(f"score prompt {prompt_id!r} does not contain eight records")
        score_prompt_ids.add(prompt_id)
        for record in records:
            text = record.get("text")
            if record.get("prompt_id") != prompt_id or not isinstance(text, str):
                raise ValueError(f"score prompt {prompt_id!r} has inconsistent records")
            key = (prompt_id, text)
            if key in score_by_text:
                raise ValueError(f"score prompt {prompt_id!r} has duplicate texts")
            score_by_text[key] = record
    if score_prompt_ids != prompt_ids:
        raise ValueError("score, clustered-generation, and target prompt IDs differ")

    generation_by_text = {
        (generation.prompt_id, generation.text): generation for generation in generation_tuple
    }
    views: dict[str, dict[str, float | int]] = {}
    for view in ("random1", "best1", "mode1", "diverse4", "all8"):
        selected_generations: list[TeacherGeneration] = []
        selected_scores: list[Mapping[str, Any]] = []
        cluster_counts: list[int] = []
        for prompt_id in sorted(prompt_ids):
            prompt_views = raw_targets[prompt_id]
            if not isinstance(prompt_views, Mapping):
                raise ValueError(f"invalid target views for {prompt_id!r}")
            texts = prompt_views.get(view)
            if not isinstance(texts, list) or not texts:
                raise ValueError(f"missing target view {view!r} for {prompt_id!r}")
            current = [generation_by_text[(prompt_id, str(text))] for text in texts]
            selected_generations.extend(current)
            selected_scores.extend(score_by_text[(prompt_id, str(text))] for text in texts)
            cluster_counts.append(len({generation.cluster_id for generation in current}))
        views[view] = {
            "num_responses": len(selected_generations),
            "quality_mean": statistics.fmean(
                generation.quality_score for generation in selected_generations
            ),
            "length_stop_rate": sum(
                record.get("finish_reason") == "length" for record in selected_scores
            )
            / len(selected_scores),
            "completion_tokens_mean": statistics.fmean(
                int(record["completion_tokens"]) for record in selected_scores
            ),
            "primary_clusters_per_prompt_mean": statistics.fmean(cluster_counts),
            "four_primary_clusters_rate": sum(value == 4 for value in cluster_counts)
            / len(cluster_counts),
        }

    overlaps: dict[str, dict[str, float | int]] = {}
    for name, left, right in (
        ("random1_equals_best1", "random1", "best1"),
        ("mode1_equals_best1", "mode1", "best1"),
        ("random1_equals_mode1", "random1", "mode1"),
    ):
        count = sum(
            raw_targets[prompt_id][left] == raw_targets[prompt_id][right]
            for prompt_id in prompt_ids
        )
        overlaps[name] = {"count": count, "rate": count / len(prompt_ids)}

    diagnostics = cluster_metadata.get("prompt_diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != prompt_ids:
        raise ValueError("cluster metadata prompt diagnostics do not align")
    threshold_names: set[str] | None = None
    threshold_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for prompt_id in sorted(prompt_ids):
        prompt = diagnostics[prompt_id]
        if not isinstance(prompt, Mapping):
            raise ValueError(f"invalid cluster diagnostics for {prompt_id!r}")
        raw = prompt.get("raw")
        instructed = prompt.get("instructed")
        if not isinstance(raw, Mapping) or not isinstance(instructed, Mapping):
            raise ValueError(f"incomplete cluster diagnostics for {prompt_id!r}")
        raw_counts = raw.get("clusters_by_threshold")
        instructed_counts = instructed.get("clusters_by_threshold")
        if not isinstance(raw_counts, Mapping) or not isinstance(instructed_counts, Mapping):
            raise ValueError(f"missing threshold cluster counts for {prompt_id!r}")
        current_names = set(raw_counts)
        if current_names != set(instructed_counts):
            raise ValueError(f"raw and instructed thresholds differ for {prompt_id!r}")
        if threshold_names is None:
            threshold_names = current_names
        elif current_names != threshold_names:
            raise ValueError("cluster thresholds changed across prompts")
        for threshold in current_names:
            threshold_rows[str(threshold)].append(
                (int(raw_counts[threshold]), int(instructed_counts[threshold]))
            )
    thresholds = {
        threshold: {
            "raw_cluster_mean": statistics.fmean(raw for raw, _ in rows),
            "raw_cluster_median": statistics.median(raw for raw, _ in rows),
            "instructed_cluster_mean": statistics.fmean(value for _, value in rows),
            "instructed_cluster_median": statistics.median(value for _, value in rows),
            "instructed_greater_count": sum(value > raw for raw, value in rows),
            "equal_count": sum(value == raw for raw, value in rows),
            "instructed_lower_count": sum(value < raw for raw, value in rows),
        }
        for threshold, rows in sorted(threshold_rows.items(), key=lambda item: float(item[0]))
    }
    primary_counts = [
        len({item.cluster_id for item in grouped[prompt_id]}) for prompt_id in prompt_ids
    ]
    return {
        "num_prompts": len(prompt_ids),
        "num_generations": len(generation_tuple),
        "primary_cluster_count_mean": statistics.fmean(primary_counts),
        "primary_cluster_count_median": statistics.median(primary_counts),
        "primary_cluster_count_distribution": {
            str(value): count for value, count in sorted(Counter(primary_counts).items())
        },
        "views": views,
        "single_view_overlap": overlaps,
        "thresholds": thresholds,
    }


def render_teacher_target_markdown(summary: Mapping[str, Any]) -> str:
    """Render teacher-view and clustering sensitivity findings with explicit caveats."""

    lines = [
        "# Frozen teacher-target findings",
        "",
        (
            f"The artifact contains {int(summary['num_prompts']):,} prompts and "
            f"{int(summary['num_generations']):,} frozen teacher samples. Cluster counts are "
            "fixed embedding-based operational modes, not expert novelty labels."
        ),
        "",
        (
            "Primary instructed clusters per prompt (mean / median): "
            f"{_number(summary['primary_cluster_count_mean'])} / "
            f"{_number(summary['primary_cluster_count_median'])}."
        ),
        "",
        "## Derived views",
        "",
        (
            "| View | Responses | Quality mean | Length-stop rate | "
            "Clusters/prompt | Four-cluster prompts |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view, values in summary["views"].items():
        lines.append(
            f"| `{view}` | {int(values['num_responses'])} | "
            f"{_number(values['quality_mean'])} | {_number(values['length_stop_rate'])} | "
            f"{_number(values['primary_clusters_per_prompt_mean'])} | "
            f"{_number(values['four_primary_clusters_rate'])} |"
        )
    lines.extend(["", "## Single-view overlap", ""])
    for name, values in summary["single_view_overlap"].items():
        lines.append(
            f"- `{name}`: {int(values['count']):,}/{int(summary['num_prompts']):,} "
            f"({_number(values['rate'])})."
        )
    lines.extend(
        [
            "",
            "## Embedding-instruction sensitivity",
            "",
            (
                "| Threshold | Raw mean | Instructed mean | "
                "Instructed greater / equal / lower prompts |"
            ),
            "|---:|---:|---:|---:|",
        ]
    )
    for threshold, values in summary["thresholds"].items():
        lines.append(
            f"| {threshold} | {_number(values['raw_cluster_mean'])} | "
            f"{_number(values['instructed_cluster_mean'])} | "
            f"{int(values['instructed_greater_count'])} / {int(values['equal_count'])} / "
            f"{int(values['instructed_lower_count'])} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _number(value: Any) -> str:
    return f"{float(value):.6g}"
