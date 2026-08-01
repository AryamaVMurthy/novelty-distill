#!/usr/bin/env python3
"""Run one bounded baseline through the pinned official OPSD checkout."""

import argparse
import json
from pathlib import Path

from novelty_distill.training.opsd import execute_opsd_training, load_opsd_run_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/baselines.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("third_party/manifest.yaml"))
    parser.add_argument("--official-root", type=Path, required=True)
    args = parser.parse_args()

    metadata = execute_opsd_training(
        load_opsd_run_spec(args.config),
        scratch_root=args.scratch_root,
        registry_path=args.registry,
        manifest_path=args.manifest,
        official_root=args.official_root,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
