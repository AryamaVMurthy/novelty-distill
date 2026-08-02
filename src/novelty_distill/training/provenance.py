"""Content-addressed provenance fields shared by training backends."""

import hashlib
from pathlib import Path


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
