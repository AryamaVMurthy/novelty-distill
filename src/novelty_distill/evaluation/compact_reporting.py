"""Dependency-free tables and SVG plots for the compact baseline study."""

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any


def _summary_value(summary: Mapping[str, Any], metric: str) -> float:
    return float(summary[f"{metric}_mean"])


def _render_svg(
    *,
    title: str,
    panels: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    width = 960
    height = 460
    panel_width = 420
    plot_top = 90
    plot_height = 270
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:g}" y="34" text-anchor="middle" '
        f'font-family="sans-serif" font-size="22">{escape(title)}</text>',
    ]
    if any(
        summaries[method].get("evidence_status") == "quarantined" for method in methods
    ):
        lines.append(
            '<text x="480" y="450" text-anchor="middle" font-family="sans-serif" '
            'font-size="11">† quarantined; descriptive only</text>'
        )
    for panel_index, panel in enumerate(panels):
        metric = str(panel["metric"])
        label = str(panel["label"])
        maximum = float(panel["maximum"])
        if maximum <= 0:
            raise ValueError(f"plot maximum must be positive for {metric}")
        left = 65 + panel_index * 460
        baseline = plot_top + plot_height
        lines.extend(
            [
                f'<text x="{left + panel_width / 2:g}" y="68" text-anchor="middle" '
                f'font-family="sans-serif" font-size="16">{escape(label)}</text>',
                f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{baseline}" '
                'stroke="#444"/>',
                f'<line x1="{left}" y1="{baseline}" x2="{left + panel_width}" '
                f'y2="{baseline}" stroke="#444"/>',
                f'<text x="{left - 8}" y="{plot_top + 5}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11">{maximum:g}</text>',
                f'<text x="{left - 8}" y="{baseline + 4}" text-anchor="end" '
                'font-family="sans-serif" font-size="11">0</text>',
            ]
        )
        slot = panel_width / len(methods)
        bar_width = min(46.0, slot * 0.62)
        for method_index, method in enumerate(methods):
            value = _summary_value(summaries[method], metric)
            bounded = min(max(value, 0.0), maximum)
            bar_height = plot_height * bounded / maximum
            x = left + method_index * slot + (slot - bar_width) / 2
            y = baseline - bar_height
            color = colors[method_index % len(colors)]
            display_method = (
                f"{method}†"
                if summaries[method].get("evidence_status") == "quarantined"
                else method
            )
            lines.extend(
                [
                    f'<rect x="{x:g}" y="{y:g}" width="{bar_width:g}" '
                    f'height="{bar_height:g}" fill="{color}"/>',
                    f'<text x="{x + bar_width / 2:g}" y="{y - 6:g}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="10">{value:.3f}</text>',
                    f'<text x="{x + bar_width / 2:g}" y="{baseline + 18}" '
                    f'text-anchor="middle" font-family="sans-serif" font-size="10">'
                    f'{escape(display_method)}</text>',
                ]
            )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_compact_artifact_bundle(
    *, payload: Mapping[str, Any], config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Write the declared summary table and exactly two deterministic SVG plots."""

    raw_summaries = payload.get("method_summaries")
    if not isinstance(raw_summaries, Mapping):
        raise ValueError("compact report requires method summaries")
    summaries = {
        str(method): summary
        for method, summary in raw_summaries.items()
        if isinstance(summary, Mapping)
    }
    methods = tuple(str(method) for method in config["method_order"])
    if set(methods) != set(summaries) or len(methods) != len(summaries):
        raise ValueError("compact method order must exactly match method summaries")
    metrics = tuple(str(metric) for metric in config["table_metrics"])
    plots = tuple(config["plots"])
    if len(plots) != 2:
        raise ValueError("compact artifact bundle requires exactly two plots")

    output_dir.mkdir(parents=True, exist_ok=True)
    table = io.StringIO()
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(("method", "evidence_status", "evidence_note", "n", *metrics))
    for method in methods:
        summary = summaries[method]
        writer.writerow(
            (
                method,
                str(summary.get("evidence_status", "unspecified")),
                str(summary.get("evidence_note", "")),
                int(summary["n"]),
                *(_summary_value(summary, metric) for metric in metrics),
            )
        )
    table_name = "baseline-summary.csv"
    (output_dir / table_name).write_text(table.getvalue(), encoding="utf-8")

    filenames = [table_name]
    for plot in plots:
        filename = str(plot["filename"])
        if Path(filename).name != filename or not filename.endswith(".svg"):
            raise ValueError("compact plot filenames must be path-safe SVG names")
        rendered = _render_svg(
            title=str(plot["title"]),
            panels=tuple(plot["panels"]),
            methods=methods,
            summaries=summaries,
        )
        (output_dir / filename).write_text(rendered, encoding="utf-8")
        filenames.append(filename)

    manifest = {
        "schema_version": 1,
        "files": filenames,
        "sha256": {
            filename: hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
            for filename in filenames
        },
    }
    (output_dir / "artifact-bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
