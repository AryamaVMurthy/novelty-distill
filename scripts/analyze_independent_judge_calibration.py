#!/usr/bin/env python3
"""Analyze agreement, repeat reliability, and method means for independent judgments."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.independent_judge import (
    CalibrationEntry,
    ExternalJudgeScore,
    analyze_independent_judgments,
    independent_judge_protocol_hash,
)


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
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet_bytes = args.packet.read_bytes()
    packet = json.loads(packet_bytes)
    protocol_hash = independent_judge_protocol_hash()
    if packet.get("protocol_hash") != protocol_hash:
        raise ValueError("calibration packet protocol hash does not match this checkout")
    entries = [CalibrationEntry.model_validate(item) for item in packet["entries"]]
    ratings: dict[str, ExternalJudgeScore] = {}
    response_hashes: dict[str, str] = {}
    provider_models: set[str] = set()
    for entry in entries:
        path = args.responses / f"{entry.blind_id}.json"
        raw = path.read_bytes()
        payload = json.loads(raw)
        if (
            payload.get("blind_id") != entry.blind_id
            or payload.get("text_sha256") != entry.text_sha256
            or payload.get("protocol_hash") != protocol_hash
        ):
            raise ValueError(f"response identity mismatch at {path}")
        ratings[entry.blind_id] = ExternalJudgeScore.model_validate(payload["parsed_score"])
        response_hashes[entry.blind_id] = hashlib.sha256(raw).hexdigest()
        provider_models.add(str(payload["provider_model"]))

    summary = analyze_independent_judgments(entries=entries, ratings=ratings)
    summary["provenance"] = {
        "packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "protocol_hash": protocol_hash,
        "provider_models": sorted(provider_models),
        "response_hashes": response_hashes,
    }
    _atomic_json(args.output, summary)
    print(json.dumps({"output": str(args.output), "counts": summary["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
