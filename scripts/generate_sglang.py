#!/usr/bin/env python3
"""Generate resumable prompt shards from an official SGLang server."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from novelty_distill.generation.sglang import (
    GenerationSpec,
    Prompt,
    generate_prompt,
    load_prompts,
    pending_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    with args.config.open(encoding="utf-8") as handle:
        spec = GenerationSpec.model_validate(yaml.safe_load(handle))

    prompts = load_prompts(args.input)
    remaining = pending_prompts(prompts, args.output_dir, spec)
    print(json.dumps({"total": len(prompts), "pending": len(remaining)}, sort_keys=True))

    def run(prompt: Prompt) -> str:
        generate_prompt(
            prompt,
            spec,
            args.output_dir,
            base_url=args.base_url,
            timeout=args.timeout,
        )
        return prompt.id

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for completed, _prompt_id in enumerate(executor.map(run, remaining), start=1):
            if completed % 10 == 0 or completed == len(remaining):
                print(json.dumps({"completed": completed, "pending_at_start": len(remaining)}))


if __name__ == "__main__":
    main()
