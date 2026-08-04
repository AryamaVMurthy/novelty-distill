#!/usr/bin/env python3
"""Write a canonical summary from one validated official HypoSpace result."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from novelty_distill.evaluation.official_results import (
    OfficialEvaluationSummary,
    hypospace_metrics,
)
from novelty_distill.training.provenance import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--domain", choices=("causal", "3d", "boolean"), required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--num-generations", type=int, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = OfficialEvaluationSummary(
        eval_id=args.eval_id,
        suite="hypospace",
        domain=args.domain,
        model_identity=args.model_identity,
        expected_samples=args.expected_samples,
        num_generations=args.num_generations,
        artifact=os.path.relpath(args.input.resolve(), args.output.parent.resolve()),
        artifact_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        metrics=hypospace_metrics(
            payload,
            expected_samples=args.expected_samples,
            num_generations=args.num_generations,
        ),
    )
    atomic_json(args.output, summary.model_dump(mode="json"))


if __name__ == "__main__":
    main()
