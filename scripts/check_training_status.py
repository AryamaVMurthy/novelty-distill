#!/usr/bin/env python3
"""Exit successfully only when a declared adapter training run is complete."""

import argparse
import json
from pathlib import Path

from novelty_distill.training.status import adapter_run_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    complete = adapter_run_complete(
        args.metadata, baseline_id=args.baseline_id, max_steps=args.max_steps
    )
    print(json.dumps({"complete": complete, "metadata": str(args.metadata)}, sort_keys=True))
    if not complete:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
