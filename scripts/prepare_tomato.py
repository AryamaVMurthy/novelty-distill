#!/usr/bin/env python3
"""Prepare a fixed canonical subset from the pinned official TOMATO-Star release."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from novelty_distill.data.tomato import (
    TOMATO_DATASET_ID,
    TOMATO_REVISION,
    prepare_tomato_record,
    select_deterministic_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--task", choices=("open", "composition"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    from datasets import load_dataset

    dataset = load_dataset(
        TOMATO_DATASET_ID,
        revision=TOMATO_REVISION,
        split=args.split,
    )
    id_rows = ({"source_id": source_id} for source_id in dataset["source_id"])
    selected = select_deterministic_subset(id_rows, size=args.size, seed=args.seed)
    selected_ids = {str(row["source_id"]) for row in selected}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for raw in dataset:
                if raw["source_id"] not in selected_ids:
                    continue
                example = prepare_tomato_record(raw, split=args.split, task=args.task)
                handle.write(example.model_dump_json())
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)

    metadata = {
        "dataset": TOMATO_DATASET_ID,
        "revision": TOMATO_REVISION,
        "split": args.split,
        "task": args.task,
        "size": args.size,
        "seed": args.seed,
    }
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
