#!/usr/bin/env python3
"""Write an explicit canonical-ID subset for deterministic stress tests."""

import argparse
import os
import tempfile
from pathlib import Path

from novelty_distill.training.length_audit import select_canonical_json_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = select_canonical_json_lines(
        args.input.read_text(encoding="utf-8").splitlines(), ids=args.id
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write("\n".join(selected) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f"selected={len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
