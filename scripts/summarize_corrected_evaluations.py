#!/usr/bin/env python3
"""Freeze compact provenance and threshold curves from corrected evaluation reruns."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.audit_summary import summarize_corrected_evaluations


def _parse_pairs(values: list[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        method, separator, item = value.partition("=")
        if not separator or not method.strip() or not item.strip() or method in parsed:
            raise ValueError(f"invalid or duplicate METHOD={label} value {value!r}")
        parsed[method] = item
    return parsed


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
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
    parser.add_argument("--input", action="append", required=True, metavar="METHOD=PATH")
    parser.add_argument("--job", action="append", required=True, metavar="METHOD=JOB_ID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_inputs = _parse_pairs(args.input, label="PATH")
    jobs = _parse_pairs(args.job, label="JOB_ID")
    paths = {method: Path(path) for method, path in raw_inputs.items()}
    payloads = {
        method: json.loads(path.read_text(encoding="utf-8"))
        for method, path in paths.items()
    }
    result = summarize_corrected_evaluations(
        payloads=payloads,
        input_sha256={
            method: hashlib.sha256(path.read_bytes()).hexdigest()
            for method, path in paths.items()
        },
        jobs=jobs,
    )
    result["input_paths"] = {
        method: str(path.resolve()) for method, path in sorted(paths.items())
    }
    _atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "methods": sorted(payloads)}))


if __name__ == "__main__":
    main()
