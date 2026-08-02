#!/usr/bin/env python3
"""Create resumable A3 generation shards from canonical historical hypotheses."""

import argparse
import json
from pathlib import Path

import yaml

from novelty_distill.evaluation.human_control import build_human_control_record
from novelty_distill.generation.sglang import (
    GenerationSpec,
    Prompt,
    ensure_generation_run_manifest,
    load_prompt_shard,
    prompt_shard_path,
    write_prompt_shard,
)
from novelty_distill.training.trl import load_canonical_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        example_count = sum(bool(line.strip()) for line in handle)
    examples = load_canonical_examples(args.input, limit=example_count)
    spec = GenerationSpec.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    prompts = tuple(Prompt(id=item.id, text=item.student_prompt) for item in examples)
    ensure_generation_run_manifest(
        input_path=args.input,
        output_dir=args.output_dir,
        prompts=prompts,
        spec=spec,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
    )
    completed = 0
    for example in examples:
        path = prompt_shard_path(args.output_dir, example.id)
        if path.exists():
            load_prompt_shard(path, spec)
        else:
            write_prompt_shard(
                args.output_dir,
                (build_human_control_record(example, spec=spec, tokenizer=tokenizer),),
                spec,
            )
        completed += 1
    print(json.dumps({"completed": completed, "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
