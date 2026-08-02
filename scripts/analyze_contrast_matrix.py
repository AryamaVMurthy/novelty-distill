#!/usr/bin/env python3
"""Run the frozen prompt-paired contrast family over collected model metrics."""

import argparse
import json
from pathlib import Path

import yaml

from novelty_distill.evaluation.contrasts import (
    analyze_contrasts,
    analyze_threshold_directions,
    summarize_method_metrics,
)
from novelty_distill.evaluation.reporting import render_contrast_markdown
from novelty_distill.provenance import repository_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold-input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    git_commit = repository_commit(Path(__file__).resolve().parents[1])
    rows = tuple(
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    threshold_rows = tuple(
        json.loads(line)
        for line in args.threshold_input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported contrast configuration schema")
    results = analyze_contrasts(
        rows=rows,
        contrasts=tuple(config["contrasts"]),
        metrics=tuple(config["metrics"]),
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    threshold_directions = analyze_threshold_directions(
        rows=threshold_rows,
        contrasts=tuple(config["contrasts"]),
        metric_directions=dict(config["threshold_metric_directions"]),
    )
    payload = {
        "schema_version": 2,
        "git_commit": git_commit,
        "bootstrap_samples": config["bootstrap_samples"],
        "seed": config["seed"],
        "holm_family": "all declared contrasts within each metric",
        "method_summaries": summarize_method_metrics(
            rows=rows,
            metrics=tuple(config.get("descriptive_metrics", config["metrics"])),
        ),
        "results": results,
        "threshold_direction_counts": threshold_directions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_contrast_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": len(results),
                "threshold_metrics": len(threshold_directions),
                "output": str(args.output),
                "markdown": str(args.markdown),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
