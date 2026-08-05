#!/usr/bin/env python3
"""Rebind a frozen judge sample to a repaired, content-hashed judge protocol."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.independent_judge import (
    CalibrationEntry,
    independent_judge_protocol_hash,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-old-protocol-hash", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    if len(args.expected_old_protocol_hash) != 64 or not args.reason.strip():
        raise ValueError("a 64-character old protocol hash and non-empty reason are required")

    source_bytes = args.input.read_bytes()
    source = json.loads(source_bytes)
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != 1
        or source.get("protocol_hash") != args.expected_old_protocol_hash
        or "protocol_migration" in source
    ):
        raise ValueError("input is not the expected unmigrated judge packet")
    entries = [CalibrationEntry.model_validate(item) for item in source.get("entries", ())]
    if not entries or len({entry.blind_id for entry in entries}) != len(entries):
        raise ValueError("input packet entries must be non-empty with unique blind IDs")

    new_protocol_hash = independent_judge_protocol_hash()
    if new_protocol_hash == args.expected_old_protocol_hash:
        raise ValueError("new judge protocol is identical to the old protocol")
    entry_bytes = json.dumps(source["entries"], sort_keys=True, separators=(",", ":")).encode()
    migrated = {
        **source,
        "protocol_hash": new_protocol_hash,
        "protocol_migration": {
            "old_protocol_hash": args.expected_old_protocol_hash,
            "new_protocol_hash": new_protocol_hash,
            "reason": args.reason.strip(),
            "source_packet_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "frozen_entries_sha256": hashlib.sha256(entry_bytes).hexdigest(),
        },
    }
    _atomic_json(args.output, migrated)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "entries": len(entries),
                "protocol_hash": new_protocol_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
