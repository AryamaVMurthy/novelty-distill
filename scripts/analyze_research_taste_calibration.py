#!/usr/bin/env python3
"""Validate blinded human labels and evaluate the automatic taste annotator gate."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.taste_calibration import (
    analyze_human_taste_agreement,
    normalize_human_packet_row,
)
from novelty_distill.provenance import repository_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--human", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _parse_humans(values: list[str]) -> dict[str, Path]:
    humans: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip() or name in humans:
            raise ValueError(f"invalid or duplicate NAME=JSONL input {value!r}")
        humans[name] = Path(raw_path)
    return humans


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"human label row is not an object: {path}")
                records.append(normalize_human_packet_row(payload))
    if not records:
        raise ValueError(f"human label file is empty: {path}")
    return tuple(records)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Research-taste human-agreement gate",
        "",
        f"Gate result: **{'PASS' if result['passed'] else 'FAIL'}**.",
        "",
        "| Axis | Comparison | Cohen's kappa |",
        "|---|---|---:|",
    ]
    for family in ("automated_vs_human", "human_vs_human"):
        for axis, comparisons in result[family].items():
            for name, value in comparisons.items():
                lines.append(f"| {axis} | {family}.{name} | {value:.4f} |")
    if result["failures"]:
        lines.extend(["", "Failed checks:", ""])
        lines.extend(f"- `{failure}`" for failure in result["failures"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = config["human_validation"]
    key_payload = json.loads(args.key.read_text(encoding="utf-8"))
    humans = _parse_humans(args.human)
    result = analyze_human_taste_agreement(
        key=key_payload["key"],
        humans={name: _read_jsonl(path) for name, path in humans.items()},
        minimum_records=gate["minimum_records"],
        minimum_annotators=gate["minimum_annotators"],
        minimum_kappa=gate["minimum_cohen_kappa"],
    )
    result.update(
        {
            "repository_commit": repository_commit(Path(__file__).resolve().parents[1]),
            "key_sha256": hashlib.sha256(args.key.read_bytes()).hexdigest(),
            "human_inputs": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in sorted(humans.items())
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(_markdown(result), encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
