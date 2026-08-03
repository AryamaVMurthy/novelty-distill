#!/usr/bin/env python3
"""Collect complete official suites into a seed-balanced descriptive matrix."""

import argparse
import json
from pathlib import Path

from novelty_distill.evaluation.official_matrix import (
    aggregate_official_matrix,
    render_official_matrix_markdown,
)
from novelty_distill.provenance import repository_commit
from novelty_distill.training.provenance import atomic_json


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item for item in value.split(",") if item)
    if not result:
        raise ValueError("comma-separated value cannot be empty")
    return result


def _inputs(values: list[str]) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for value in values:
        identity, separator, raw_path = value.partition("=")
        method, seed_separator, raw_seed = identity.partition(":")
        if not separator or not seed_separator or not method or not raw_seed or not raw_path:
            raise ValueError("each --input must be METHOD:SEED=PATH")
        key = (method, int(raw_seed))
        if key in result:
            raise ValueError(f"duplicate official input {identity!r}")
        result[key] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    result = aggregate_official_matrix(
        _inputs(args.input),
        expected_methods=_csv(args.methods),
        expected_seeds=tuple(int(seed) for seed in _csv(args.seeds)),
    )
    result["repository_commit"] = repository_commit(Path(__file__).resolve().parents[1])
    atomic_json(args.output, result)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_official_matrix_markdown(result), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
