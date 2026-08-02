#!/usr/bin/env python3
"""Seed a larger generation run from a compatible completed subset."""

import argparse
import json
from pathlib import Path

import yaml

from novelty_distill.generation.sglang import (
    GenerationSpec,
    bootstrap_generation_shards,
    ensure_generation_run_manifest,
    load_prompts,
    pending_prompts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--target-input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--served-artifact-identity")
    args = parser.parse_args()
    spec = GenerationSpec.model_validate(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    prompts = load_prompts(args.target_input)
    ensure_generation_run_manifest(
        input_path=args.target_input,
        output_dir=args.target_dir,
        prompts=prompts,
        spec=spec,
        served_artifact_identity=args.served_artifact_identity,
    )
    copied, reused = bootstrap_generation_shards(
        source_dir=args.source_dir, target_dir=args.target_dir
    )
    remaining = pending_prompts(prompts, args.target_dir, spec)
    print(
        json.dumps(
            {
                "copied": copied,
                "reused": reused,
                "pending": len(remaining),
                "total": len(prompts),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
