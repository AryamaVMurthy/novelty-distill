#!/usr/bin/env python3
"""Analyze secondary research-taste distributions across aligned methods."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.research_taste import (
    ResearchTasteRecord,
    analyze_research_taste_matrix,
    render_research_taste_markdown,
    research_taste_protocol_hash,
)
from novelty_distill.evaluation.taste_shards import load_research_taste_shard
from novelty_distill.provenance import repository_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="METHOD=DIR")
    parser.add_argument("--human-method", default="A3")
    parser.add_argument("--teacher-method", default="A1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"research-taste input must be METHOD=DIR, got {value!r}")
        method, raw_path = value.split("=", 1)
        if not method.strip() or method in inputs:
            raise ValueError(f"invalid or duplicate research-taste method {method!r}")
        inputs[method] = Path(raw_path)
    return inputs


def _load_records(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    shard_paths = sorted(path.glob("*.json"))
    if not shard_paths:
        raise ValueError(f"no research-taste shards found in {path}")
    for shard_path in shard_paths:
        preview = json.loads(shard_path.read_text(encoding="utf-8"))
        raw_records = preview.get("records") if isinstance(preview, dict) else None
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"research-taste shard has no records: {shard_path}")
        payload = load_research_taste_shard(
            shard_path, samples_per_prompt=len(raw_records)
        )
        for raw_record in payload["records"]:
            record = ResearchTasteRecord.model_validate(
                {
                    key: value
                    for key, value in raw_record.items()
                    if key
                    not in {
                        "text",
                        "request_id",
                        "model",
                    }
                }
            )
            records.append(record.model_dump(mode="json"))
    return tuple(records)


def _tree_provenance(path: Path) -> dict[str, Any]:
    shard_paths = sorted(path.glob("*.json"), key=lambda candidate: candidate.name)
    if not shard_paths:
        raise ValueError(f"no research-taste shards found in {path}")
    digest = hashlib.sha256()
    total_bytes = 0
    for shard_path in shard_paths:
        content = shard_path.read_bytes()
        total_bytes += len(content)
        digest.update(shard_path.name.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return {
        "path": str(path.resolve()),
        "num_shards": len(shard_paths),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "secondary_descriptive":
        raise ValueError("research-taste analysis must remain secondary_descriptive")
    inputs = _parse_inputs(args.input)
    bootstrap = config["bootstrap"]
    result = analyze_research_taste_matrix(
        {method: _load_records(path) for method, path in inputs.items()},
        human_method=args.human_method,
        teacher_method=args.teacher_method,
        bootstrap_resamples=bootstrap["resamples"],
        bootstrap_seed=bootstrap["seed"],
        bootstrap_confidence_level=bootstrap["confidence_level"],
    )
    result["source"] = config["source"]
    result["protocol_hash"] = research_taste_protocol_hash()
    result["repository_commit"] = repository_commit(Path(__file__).resolve().parents[1])
    result["inputs"] = {
        method: _tree_provenance(path) for method, path in sorted(inputs.items())
    }
    result["human_validation"] = config["human_validation"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(
        render_research_taste_markdown(result, config), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
