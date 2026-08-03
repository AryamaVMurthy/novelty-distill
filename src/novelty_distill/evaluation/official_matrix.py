"""Balanced descriptive aggregation for complete official benchmark suites."""

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def aggregate_official_matrix(
    inputs: Mapping[tuple[str, int], Path],
    *,
    expected_methods: Sequence[str],
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Validate and aggregate an exact method-by-seed matrix without pooling runs."""

    methods = tuple(expected_methods)
    seeds = tuple(expected_seeds)
    if (
        not methods
        or not seeds
        or len(methods) != len(set(methods))
        or len(seeds) != len(set(seeds))
        or any(not method for method in methods)
    ):
        raise ValueError("expected methods and seeds must be non-empty and unique")
    expected = {(method, seed) for method in methods for seed in seeds}
    if set(inputs) != expected:
        raise ValueError("official inputs are not the complete method-by-seed grid")

    runs: list[dict[str, Any]] = []
    metric_names: tuple[str, ...] | None = None
    values: dict[str, dict[str, list[float]]] = {
        method: {} for method in methods
    }
    for method in methods:
        for seed in seeds:
            path = inputs[(method, seed)].resolve(strict=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError(f"invalid combined official result: {path}")
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError(f"official result has no metrics: {path}")
            observed_names = tuple(sorted(metrics))
            if metric_names is None:
                metric_names = observed_names
            elif observed_names != metric_names:
                raise ValueError("official result metric sets differ")
            normalized: dict[str, float] = {}
            for name in observed_names:
                value = metrics[name]
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError(f"official metric {name!r} is not numeric: {path}")
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"official metric {name!r} is not finite: {path}")
                normalized[name] = numeric
                values[method].setdefault(name, []).append(numeric)
            eval_id = payload.get("eval_id")
            identity = payload.get("model_identity")
            if (
                not isinstance(eval_id, str)
                or not eval_id
                or not isinstance(identity, str)
                or not identity
            ):
                raise ValueError(f"official result has no evaluation/model identity: {path}")
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "eval_id": eval_id,
                    "model_identity": identity,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "metrics": normalized,
                }
            )
    assert metric_names is not None
    summaries = {
        method: {
            name: {
                "mean": statistics.fmean(metric_values),
                "population_sd": statistics.pstdev(metric_values),
                "seed_count": len(metric_values),
            }
            for name, metric_values in sorted(values[method].items())
        }
        for method in methods
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "methods": list(methods),
        "seeds": list(seeds),
        "metric_names": list(metric_names),
        "interpretation": "descriptive_seed_balanced_official_benchmarks",
        "method_summaries": summaries,
        "runs": runs,
    }


def render_official_matrix_markdown(payload: Mapping[str, Any]) -> str:
    """Render compact seed-balanced benchmark means and population SDs."""

    lines = [
        "# Official benchmark matrix",
        "",
        "Descriptive mean ± population SD across the declared training seeds.",
        "",
        "| Method | Metric | Mean | Population SD | Seeds |",
        "|---|---|---:|---:|---:|",
    ]
    summaries = payload["method_summaries"]
    for method in payload["methods"]:
        for metric, summary in summaries[method].items():
            lines.append(
                f"| {method} | {metric} | {summary['mean']:.6f} | "
                f"{summary['population_sd']:.6f} | {summary['seed_count']} |"
            )
    return "\n".join(lines) + "\n"
