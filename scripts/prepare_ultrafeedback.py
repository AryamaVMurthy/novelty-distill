#!/usr/bin/env python3
"""Prepare a fixed 1k UltraFeedback pilot and its reusable four-response views."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from novelty_distill.data.ultrafeedback import (
    ULTRAFEEDBACK_DATASET_ID,
    ULTRAFEEDBACK_REVISION,
    prepare_ultrafeedback_record,
    ultrafeedback_record_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    destinations = (args.output, args.targets, args.manifest)
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    if args.size <= 0:
        raise ValueError("size must be positive")

    from datasets import load_dataset

    dataset = load_dataset(
        ULTRAFEEDBACK_DATASET_ID,
        revision=ULTRAFEEDBACK_REVISION,
        split="train",
    )
    identifiers = [ultrafeedback_record_id(row) for row in dataset]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("UltraFeedback content identities must be unique for stable pilot IDs")
    if args.size > len(identifiers):
        raise ValueError(f"requested {args.size} rows from {len(identifiers)}")
    selected_indices = sorted(
        range(len(identifiers)),
        key=lambda index: (
            hashlib.sha256(f"{args.seed}\0{identifiers[index]}".encode()).digest(),
            identifiers[index],
        ),
    )[: args.size]

    rows: list[str] = []
    targets: dict[str, dict[str, list[str]]] = {}
    for index in selected_indices:
        example, views = prepare_ultrafeedback_record(dataset[index], seed=args.seed)
        rows.append(example.model_dump_json())
        targets[example.id] = views
    canonical_content = ("\n".join(rows) + "\n").encode()
    target_payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": ULTRAFEEDBACK_DATASET_ID,
        "revision": ULTRAFEEDBACK_REVISION,
        "seed": args.seed,
        "targets": targets,
    }
    target_content = (
        json.dumps(target_payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    manifest_payload = {
        "schema_version": 1,
        "dataset": ULTRAFEEDBACK_DATASET_ID,
        "revision": ULTRAFEEDBACK_REVISION,
        "split": "train",
        "size": args.size,
        "seed": args.seed,
        "canonical_sha256": hashlib.sha256(canonical_content).hexdigest(),
        "targets_sha256": hashlib.sha256(target_content).hexdigest(),
        "views": {
            "random1": 1,
            "best1": 1,
            "diverse2": 2,
            "diverse4": 4,
            "all4": 4,
        },
    }
    manifest_content = (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode()
    _atomic_bytes(args.output, canonical_content)
    _atomic_bytes(args.targets, target_content)
    _atomic_bytes(args.manifest, manifest_content)
    print(json.dumps(manifest_payload, sort_keys=True))


if __name__ == "__main__":
    main()
