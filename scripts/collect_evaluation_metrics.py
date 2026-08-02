#!/usr/bin/env python3
"""Collect aligned evaluation JSON into prompt-level JSONL for paired analysis."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from novelty_distill.evaluation.matrix import (
    collect_prompt_metric_rows,
    collect_threshold_metric_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="METHOD=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-output", type=Path)
    return parser.parse_args()


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or not method.strip() or not raw_path.strip():
            raise ValueError("each --input must be METHOD=PATH")
        if method in inputs:
            raise ValueError(f"duplicate method {method!r}")
        inputs[method] = Path(raw_path)
    return inputs


def _atomic_write_rows(
    output: Path, rows: tuple[dict[str, float | str], ...]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                json.dump(row, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    inputs = _parse_inputs(args.input)
    payloads = {
        method: json.loads(path.read_text(encoding="utf-8"))
        for method, path in inputs.items()
    }
    rows = collect_prompt_metric_rows(payloads)
    _atomic_write_rows(args.output, rows)

    manifest = {
        "schema_version": 1,
        "methods": sorted(inputs),
        "num_rows": len(rows),
        "num_prompts": len(rows) // len(inputs),
        "inputs": {
            method: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for method, path in sorted(inputs.items())
        },
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))

    if args.threshold_output is not None:
        threshold_rows = collect_threshold_metric_rows(payloads)
        _atomic_write_rows(args.threshold_output, threshold_rows)
        threshold_manifest = {
            "schema_version": 1,
            "methods": sorted(inputs),
            "num_rows": len(threshold_rows),
            "num_prompts": len({str(row["prompt_id"]) for row in threshold_rows}),
            "num_thresholds": len({str(row["threshold"]) for row in threshold_rows}),
            "inputs": manifest["inputs"],
            "output_sha256": hashlib.sha256(
                args.threshold_output.read_bytes()
            ).hexdigest(),
        }
        threshold_manifest_path = args.threshold_output.with_suffix(
            args.threshold_output.suffix + ".manifest.json"
        )
        threshold_manifest_path.write_text(
            json.dumps(threshold_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(threshold_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
