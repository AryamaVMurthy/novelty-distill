#!/usr/bin/env python3
"""Export immutable run/evaluation JSON to the pinned W&B SDK."""

import argparse
import json
import os
from pathlib import Path

from novelty_distill.tracking.wandb_export import export_wandb_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "novelty-distill"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--mode",
        choices=("offline", "online", "disabled"),
        default=os.environ.get("WANDB_MODE", "offline"),
    )
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(args.run_dir))
    os.environ.setdefault("WANDB_DATA_DIR", str(args.run_dir / "data"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(args.run_dir / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(args.run_dir / "config"))

    import wandb

    result = export_wandb_records(
        tuple(args.input),
        manifest_path=args.manifest,
        wandb_module=wandb,
        project=args.project,
        entity=args.entity,
        mode=args.mode,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
