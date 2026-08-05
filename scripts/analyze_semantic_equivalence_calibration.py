#!/usr/bin/env python3
"""Select an embedding boundary from two blinded human label files."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.semantic_calibration import (
    HumanEquivalenceLabel,
    PrivateSemanticPair,
    analyze_semantic_calibration,
    semantic_calibration_protocol_hash,
)

DEFAULT_THRESHOLD_GRID = tuple(round(0.80 + 0.01 * index, 2) for index in range(20))


def _load_labels(path: Path) -> dict[str, HumanEquivalenceLabel]:
    labels: dict[str, HumanEquivalenceLabel] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            label = HumanEquivalenceLabel.model_validate(json.loads(line))
            if label.blind_id in labels:
                raise ValueError(f"duplicate label {label.blind_id} in {path}")
            labels[label.blind_id] = label
    if not labels:
        raise ValueError(f"label file is empty: {path}")
    return labels


def _load_adjudication(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            blind_id = str(row.get("blind_id", "")).strip()
            label = str(row.get("label", "")).strip()
            if not blind_id or label not in {"equivalent", "not_equivalent"}:
                raise ValueError(f"invalid adjudication row in {path}")
            if blind_id in labels:
                raise ValueError(f"duplicate adjudication {blind_id} in {path}")
            labels[blind_id] = label
    return labels


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
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--rater-one", type=Path, required=True)
    parser.add_argument("--rater-two", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    private_bytes = args.private_key.read_bytes()
    private = json.loads(private_bytes)
    protocol_hash = semantic_calibration_protocol_hash()
    if private.get("protocol_hash") != protocol_hash:
        raise ValueError("private packet protocol hash does not match this checkout")
    entries = [PrivateSemanticPair.model_validate(item) for item in private["entries"]]
    summary = analyze_semantic_calibration(
        entries=entries,
        rater_one=_load_labels(args.rater_one),
        rater_two=_load_labels(args.rater_two),
        adjudicated=_load_adjudication(args.adjudication),
        threshold_grid=DEFAULT_THRESHOLD_GRID,
    )
    summary["provenance"] = {
        "protocol_hash": protocol_hash,
        "private_key_sha256": hashlib.sha256(private_bytes).hexdigest(),
        "rater_one_sha256": hashlib.sha256(args.rater_one.read_bytes()).hexdigest(),
        "rater_two_sha256": hashlib.sha256(args.rater_two.read_bytes()).hexdigest(),
        "adjudication_sha256": (
            hashlib.sha256(args.adjudication.read_bytes()).hexdigest()
            if args.adjudication is not None
            else None
        ),
    }
    _atomic_json(args.output, summary)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_threshold": summary["selected_threshold"],
                "inter_rater_gate_passed": summary["inter_rater"]["passed"],
            },
            sort_keys=True,
        )
    )
    if not summary["inter_rater"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
