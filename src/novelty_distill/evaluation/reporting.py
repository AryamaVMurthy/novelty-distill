"""Deterministic human-readable rendering of the frozen contrast analysis."""

from collections.abc import Mapping
from typing import Any


def render_contrast_markdown(payload: Mapping[str, Any]) -> str:
    """Render inferential estimates and descriptive threshold directions without promotion."""

    if payload.get("schema_version") != 2:
        raise ValueError("contrast report requires schema version 2")
    results = payload.get("results")
    directions = payload.get("threshold_direction_counts")
    if not isinstance(results, Mapping) or not results:
        raise ValueError("contrast report has no primary results")
    if not isinstance(directions, Mapping) or not directions:
        raise ValueError("contrast report has no threshold directions")

    lines = [
        "# Frozen TOMATO contrast findings",
        "",
        (
            "All estimates are treatment minus reference over paired prompts. Confidence intervals "
            f"and sign-flip p-values use {int(payload['bootstrap_samples']):,} Monte Carlo samples "
            f"with seed {int(payload['seed'])}; Holm correction is applied across every declared "
            "contrast within each metric. These are operational judge/embedding outcomes, not "
            "human-validated scientific novelty labels."
        ),
        "",
        "## Primary-threshold paired estimates",
        "",
    ]
    for metric, raw_contrasts in results.items():
        if not isinstance(raw_contrasts, Mapping):
            raise ValueError(f"metric {metric!r} has invalid contrast results")
        lines.extend(
            [
                f"### `{metric}`",
                "",
                "| Contrast | n | Mean difference | 95% CI | Cohen's dz | p | Holm p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for contrast_id, estimate in raw_contrasts.items():
            if not isinstance(estimate, Mapping):
                raise ValueError(f"contrast {contrast_id!r} is invalid")
            effect = estimate.get("effect_size")
            effect_text = "NA" if effect is None else _number(effect)
            lines.append(
                f"| `{contrast_id}` | {int(estimate['n'])} | "
                f"{_number(estimate['mean_difference'])} | "
                f"[{_number(estimate['ci_low'])}, {_number(estimate['ci_high'])}] | "
                f"{effect_text} | {_number(estimate['p_value'])} | "
                f"{_number(estimate['holm_p_value'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Threshold-curve prompt directions",
            "",
            (
                "F/T/U is the number of paired prompts favorable, tied, or unfavorable for the "
                "treatment at that threshold. `cluster_jsd` is favorable when lower; the other "
                "declared threshold metrics are favorable when higher. Direction counts are "
                "descriptive and are not an additional multiplicity-adjusted hypothesis family."
            ),
            "",
        ]
    )
    for metric, raw_contrasts in directions.items():
        if not isinstance(raw_contrasts, Mapping):
            raise ValueError(f"threshold metric {metric!r} is invalid")
        lines.extend(
            [
                f"### `{metric}`",
                "",
                "| Contrast | Preferred | Stable mean direction | Per-threshold F/T/U |",
                "|---|---|---|---|",
            ]
        )
        for contrast_id, summary in raw_contrasts.items():
            if not isinstance(summary, Mapping) or not isinstance(
                summary.get("thresholds"), Mapping
            ):
                raise ValueError(f"threshold contrast {contrast_id!r} is invalid")
            counts = "; ".join(
                f"{threshold}: {int(values['favorable_count'])}/"
                f"{int(values['tied_count'])}/{int(values['unfavorable_count'])}"
                for threshold, values in summary["thresholds"].items()
            )
            lines.append(
                f"| `{contrast_id}` | {summary['desirable_direction']} | "
                f"{summary['stable_mean_direction']} | {counts} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _number(value: Any) -> str:
    return f"{float(value):.6g}"
