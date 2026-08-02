#!/usr/bin/env python3
"""Validate and manifest the complete canonical TOMATO artifact family."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from novelty_distill.data.validation import validate_tomato_family


def _sized_paths(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        size_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"expected SIZE=PATH, received {value!r}")
        size = int(size_text)
        if size in parsed:
            raise ValueError(f"duplicate dataset size {size}")
        parsed[size] = Path(path_text)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-open", action="append", required=True)
    parser.add_argument("--train-composition", action="append", required=True)
    parser.add_argument("--test-open", type=Path, required=True)
    parser.add_argument("--test-composition", type=Path, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = validate_tomato_family(
        train_open=_sized_paths(args.train_open),
        train_composition=_sized_paths(args.train_composition),
        test_open=args.test_open,
        test_composition=args.test_composition,
        expected_test_size=args.test_size,
    )
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
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
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(rendered, end="")


if __name__ == "__main__":
    main()
