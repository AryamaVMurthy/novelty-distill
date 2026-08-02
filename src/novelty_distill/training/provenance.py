"""Content-addressed provenance fields shared by training backends."""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def file_provenance(path: Path) -> dict[str, str | int]:
    """Bind a training input to its resolved path, byte count, and SHA-256 digest."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"training provenance path is not a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON artifact so preemption cannot expose a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
