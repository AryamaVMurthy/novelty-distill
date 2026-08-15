#!/usr/bin/env python3
"""Rank the six semantic-seed short runs behind a validity gate."""

import argparse
import json
from pathlib import Path

from novelty_distill.evaluation.roll_gate import (
    select_validity_first,
    summarize_clustered_run,
)
from novelty_distill.training.provenance import atomic_json

RUN_IDS = (
    "SS0-C1",
    "SS0D-DIVERSE",
    "SS1-CR",
    "SS2-GSC",
    "SS3-TL",
    "SS4-GSC-TL",
)
SEEDED_IDS = {"SS2-GSC", "SS4-GSC-TL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validity-tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {
        run_id: summarize_clustered_run(
            args.cluster_root / f"teacher-generations-{run_id}-short-v1.jsonl",
            input_seed_repeats=2 if run_id in SEEDED_IDS else None,
        )
        for run_id in RUN_IDS
    }
    payload = {
        "schema_version": 1,
        "runs": summaries,
        "selection": select_validity_first(
            summaries,
            reference_id="SS0D-DIVERSE",
            validity_tolerance=args.validity_tolerance,
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
