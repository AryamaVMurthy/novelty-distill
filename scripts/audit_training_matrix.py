#!/usr/bin/env python3
"""Prove that every declared runnable baseline has comparable training provenance."""

import argparse
import json
from pathlib import Path

from novelty_distill.provenance import repository_commit
from novelty_distill.training.audit import audit_training_matrix
from novelty_distill.training.provenance import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, metavar="BASELINE=PATH")
    parser.add_argument("--baseline-id", action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for value in values:
        baseline_id, separator, raw_path = value.partition("=")
        if not separator or not baseline_id.strip() or not raw_path.strip():
            raise ValueError("each --input must be BASELINE=PATH")
        if baseline_id in inputs:
            raise ValueError(f"duplicate baseline {baseline_id!r}")
        inputs[baseline_id] = Path(raw_path)
    return inputs


def main() -> None:
    args = parse_args()
    result = audit_training_matrix(
        args.registry,
        _parse_inputs(args.input),
        required_baseline_ids=args.baseline_id,
    )
    result["repository_commit"] = repository_commit(Path(__file__).resolve().parents[1])
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
