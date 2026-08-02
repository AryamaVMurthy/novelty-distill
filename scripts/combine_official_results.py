#!/usr/bin/env python3
"""Combine one model's complete official benchmark suite."""

import argparse
import json
from pathlib import Path

from novelty_distill.evaluation.official_results import combine_official_summaries
from novelty_distill.training.provenance import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = combine_official_summaries(args.input, eval_id=args.eval_id)
    atomic_json(args.output, combined)
    print(json.dumps(combined, sort_keys=True))


if __name__ == "__main__":
    main()
