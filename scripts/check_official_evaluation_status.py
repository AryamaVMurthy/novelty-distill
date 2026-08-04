#!/usr/bin/env python3
"""Validate a canonical official-evaluation summary for cheap resume."""

import argparse
from pathlib import Path

from novelty_distill.evaluation.official_results import validate_official_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--suite", choices=("noveltybench", "hypospace"), required=True)
    parser.add_argument("--domain", choices=("causal", "3d", "boolean"))
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--num-generations", type=int, required=True)
    parser.add_argument("--novelty-base-seed", type=int)
    args = parser.parse_args()
    if validate_official_summary(
        args.summary,
        eval_id=args.eval_id,
        suite=args.suite,
        domain=args.domain,
        model_identity=args.model_identity,
        expected_samples=args.expected_samples,
        num_generations=args.num_generations,
        novelty_base_seed=args.novelty_base_seed,
    ):
        return
    raise SystemExit(3)


if __name__ == "__main__":
    main()
