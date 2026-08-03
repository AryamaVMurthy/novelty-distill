#!/usr/bin/env python3
"""Pretty-print one record from a prepared TOMATO JSONL artifact."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="prepared TOMATO JSONL file")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--id", dest="record_id", help="exact TOMATO record ID")
    selector.add_argument("--index", type=int, help="zero-based JSONL row index")
    return parser.parse_args()


def select_record(path: Path, *, record_id: str | None, index: int | None) -> dict[str, Any]:
    """Stream the artifact and return exactly one selected JSON object."""

    selected_index = 0 if record_id is None and index is None else index
    if selected_index is not None and selected_index < 0:
        raise ValueError("--index must be nonnegative")
    selected: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {row_index} is not a JSON object")
            if record_id is not None and row.get("id") == record_id:
                if selected is not None:
                    raise ValueError(f"duplicate TOMATO record ID {record_id!r}")
                selected = row
            elif selected_index is not None and row_index == selected_index:
                selected = row
                break
    if selected is None:
        selector = f"ID {record_id!r}" if record_id is not None else f"index {selected_index}"
        raise ValueError(f"no TOMATO record found for {selector}")
    return selected


def main() -> None:
    args = parse_args()
    record = select_record(args.input, record_id=args.record_id, index=args.index)
    print(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=False))


if __name__ == "__main__":
    main()
