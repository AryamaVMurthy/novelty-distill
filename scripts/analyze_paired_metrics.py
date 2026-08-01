#!/usr/bin/env python3
"""Compare baseline metrics over exactly paired prompt IDs."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.statistics import holm_adjust, paired_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_method: dict[str, dict[str, float]] = {}
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                prompt_id = str(row["prompt_id"])
                method = str(row["method"])
                value = float(row[args.metric])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid metric row at line {line_number}") from error
            if not math.isfinite(value):
                raise ValueError(f"non-finite metric at line {line_number}")
            method_values = by_method.setdefault(method, {})
            if prompt_id in method_values:
                raise ValueError(f"duplicate prompt_id {prompt_id!r} for method {method!r}")
            method_values[prompt_id] = value

    if args.reference not in by_method:
        raise ValueError(f"reference method {args.reference!r} is absent")
    comparisons = args.compare or sorted(set(by_method) - {args.reference})
    if not comparisons:
        raise ValueError("at least one comparison method is required")
    reference = by_method[args.reference]
    results: dict[str, dict[str, Any]] = {}
    raw_p_values: dict[str, float] = {}
    for method in comparisons:
        if method not in by_method:
            raise ValueError(f"comparison method {method!r} is absent")
        candidate = by_method[method]
        if candidate.keys() != reference.keys():
            raise ValueError(f"method {method!r} does not have exactly the reference prompt IDs")
        prompt_ids = sorted(reference)
        estimate = paired_bootstrap(
            reference=tuple(reference[prompt_id] for prompt_id in prompt_ids),
            treatment=tuple(candidate[prompt_id] for prompt_id in prompt_ids),
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        result = asdict(estimate)
        if not math.isfinite(result["effect_size"]):
            result["effect_size"] = None
        results[method] = result
        raw_p_values[method] = estimate.p_value

    adjusted = holm_adjust(raw_p_values)
    for method, p_value in adjusted.items():
        results[method]["holm_p_value"] = p_value
    payload = {
        "schema_version": 1,
        "metric": args.metric,
        "reference": args.reference,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "comparisons": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
