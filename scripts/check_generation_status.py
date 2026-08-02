#!/usr/bin/env python3
"""Validate a resumable generation run and report whether every prompt is complete."""

import argparse
import json
from pathlib import Path

import yaml

from novelty_distill.generation.sglang import (
    GenerationSpec,
    ensure_generation_run_manifest,
    load_prompts,
    pending_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = GenerationSpec.model_validate(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    prompts = load_prompts(args.input)
    ensure_generation_run_manifest(
        input_path=args.input,
        output_dir=args.output_dir,
        prompts=prompts,
        spec=spec,
    )
    remaining = pending_prompts(prompts, args.output_dir, spec)
    print(
        json.dumps(
            {"complete": not remaining, "pending": len(remaining), "total": len(prompts)},
            sort_keys=True,
        )
    )
    if remaining:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
