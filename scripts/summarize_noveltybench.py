#!/usr/bin/env python3
"""Write a canonical summary from one successful official Inspect evaluation log."""

import argparse
import hashlib
import os
from pathlib import Path

from inspect_ai.log import read_eval_log

from novelty_distill.evaluation.official_results import (
    NoveltyBenchSamplingProtocol,
    OfficialEvaluationSummary,
    noveltybench_metrics,
    noveltybench_sampling_diagnostics,
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
    parser.add_argument("--base-seed", type=int, required=True)
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
    log = read_eval_log(artifact)
    sampling_records = []
    for sample in log.samples or []:
        metadata = sample.metadata or {}
        observed_seeds = []
        for event in sample.events or []:
            event_payload = event.model_dump(mode="json")
            if event_payload.get("event") != "model":
                continue
            config = event_payload.get("config")
            if isinstance(config, dict) and config.get("seed") is not None:
                observed_seeds.append(int(config["seed"]))
        sampling_records.append(
            {
                "sample_id": sample.id,
                "completions": metadata.get("all_completions"),
                "declared_seeds": metadata.get("generation_seeds"),
                "observed_seeds": observed_seeds,
            }
        )
    sampling_diagnostics = noveltybench_sampling_diagnostics(
        sampling_records,
        expected_samples=args.expected_samples,
        num_generations=args.num_generations,
        base_seed=args.base_seed,
    )
    results_payload = results.model_dump(mode="json")
    summary = OfficialEvaluationSummary(
        eval_id=args.eval_id,
        suite="noveltybench",
        model_identity=args.model_identity,
        expected_samples=args.expected_samples,
        num_generations=args.num_generations,
        artifact=os.path.relpath(artifact.resolve(), args.output.parent.resolve()),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        metrics=noveltybench_metrics(results_payload, expected_samples=args.expected_samples),
        sampling_protocol=NoveltyBenchSamplingProtocol(base_seed=args.base_seed),
        sampling_diagnostics=sampling_diagnostics,
    )
    atomic_json(args.output, summary.model_dump(mode="json"))


if __name__ == "__main__":
    main()
