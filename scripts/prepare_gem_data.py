#!/usr/bin/env python3
"""Prepare exact pretokenized JSONL for the pinned official GEM trainer."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import CanonicalExample
from novelty_distill.training.gem import (
    build_gem_rows,
    load_teacher_targets,
    write_gem_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-examples", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--baseline-id", default="B4")
    parser.add_argument("--registry", type=Path, default=Path("configs/baselines.yaml"))
    args = parser.parse_args()
    if args.max_examples <= 0 or args.max_length <= 0:
        raise ValueError("max-examples and max-length must be positive")

    registry = load_baseline_registry(args.registry)
    matches = [baseline for baseline in registry.baselines if baseline.id == args.baseline_id]
    if len(matches) != 1:
        raise ValueError(f"baseline {args.baseline_id} is not uniquely defined")

    examples: list[CanonicalExample] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(CanonicalExample.model_validate_json(line))
            if len(examples) == args.max_examples:
                break
    if len(examples) != args.max_examples:
        raise ValueError(
            f"requested {args.max_examples} examples but {args.input} contains {len(examples)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    rows = build_gem_rows(
        matches[0],
        examples,
        teacher_targets=load_teacher_targets(args.teacher_targets),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    write_gem_jsonl(rows, args.output)
    print(f"{len(rows)}\t{args.output}")


if __name__ == "__main__":
    main()
