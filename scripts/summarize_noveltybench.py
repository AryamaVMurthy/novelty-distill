#!/usr/bin/env python3
"""Write a canonical summary from one successful official Inspect evaluation log."""

import argparse
import hashlib
from pathlib import Path

from inspect_ai.log import read_eval_log

from novelty_distill.evaluation.official_results import (
    OfficialEvaluationSummary,
    noveltybench_metrics,
)
from novelty_distill.training.provenance import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--num-generations", type=int, required=True)
    args = parser.parse_args()
    successful: list[tuple[Path, object]] = []
    for path in sorted(args.log_dir.glob("*.eval")):
        log = read_eval_log(path)
        if log.status == "success" and log.results is not None:
            successful.append((path, log.results))
    if len(successful) != 1:
        raise ValueError(
            f"expected exactly one successful NoveltyBench log, found {len(successful)}"
        )
    artifact, results = successful[0]
    results_payload = results.model_dump(mode="json")
    summary = OfficialEvaluationSummary(
        eval_id=args.eval_id,
        suite="noveltybench",
        model_identity=args.model_identity,
        expected_samples=args.expected_samples,
        num_generations=args.num_generations,
        artifact=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        metrics=noveltybench_metrics(results_payload, expected_samples=args.expected_samples),
    )
    atomic_json(args.output, summary.model_dump(mode="json"))


if __name__ == "__main__":
    main()
