#!/usr/bin/env python3
"""Run one bounded official TRL baseline from a validated project config."""

import argparse
import json
from pathlib import Path

from novelty_distill.training.trl import (
    execute_trl_training,
    load_trl_run_spec,
    override_trl_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/baselines.yaml"))
    parser.add_argument("--baseline-id")
    args = parser.parse_args()

    metadata = execute_trl_training(
        override_trl_baseline(load_trl_run_spec(args.config), args.baseline_id),
        scratch_root=args.scratch_root,
        registry_path=args.registry,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
