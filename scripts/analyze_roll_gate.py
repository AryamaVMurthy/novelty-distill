#!/usr/bin/env python3
"""Compare ordinary sampling with replicated Gaussian input rolls."""

import argparse
import json
from pathlib import Path

from novelty_distill.evaluation.roll_gate import compare_clustered_roll_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--seeded", type=Path, required=True)
    parser.add_argument("--input-seed-repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_clustered_roll_runs(
        ordinary_path=args.ordinary,
        seeded_path=args.seeded,
        input_seed_repeats=args.input_seed_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
