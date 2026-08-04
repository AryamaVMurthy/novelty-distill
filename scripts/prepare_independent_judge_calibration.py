#!/usr/bin/env python3
"""Build a paired, blinded calibration packet from frozen Qwen score shards."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.independent_judge import (
    build_blinded_calibration_sample,
    independent_judge_protocol_hash,
)


def _parse_methods(values: list[str]) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or not method.strip() or not raw_path.strip() or method in methods:
            raise ValueError(f"invalid or duplicate METHOD=DIR value {value!r}")
        methods[method] = Path(raw_path)
    return methods


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


def _load_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = str(row["id"])
            if prompt_id in prompts:
                raise ValueError(f"duplicate prompt {prompt_id}")
            prompts[prompt_id] = str(row["student_prompt"])
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def _load_score_dir(path: Path, *, max_sample_index: int) -> tuple[list[dict], str]:
    candidates: list[dict] = []
    digest = hashlib.sha256()
    paths = sorted(item for item in path.glob("*.json") if not item.name.startswith("_"))
    if not paths:
        raise ValueError(f"no score shards found in {path}")
    for score_path in paths:
        raw = score_path.read_bytes()
        digest.update(score_path.name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).digest())
        payload = json.loads(raw)
        prompt_id = str(payload["prompt_id"])
        for record in payload["records"]:
            sample_index = int(record["sample_index"])
            if sample_index < max_sample_index:
                if str(record["prompt_id"]) != prompt_id:
                    raise ValueError(f"record prompt mismatch in {score_path}")
                candidates.append(
                    {
                        "prompt_id": prompt_id,
                        "sample_index": sample_index,
                        "text": record["text"],
                        "dimensions": record["dimensions"],
                    }
                )
    return candidates, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True, metavar="METHOD=SCORE_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-method", type=int, default=60)
    parser.add_argument("--repeat-fraction", type=float, default=0.1)
    parser.add_argument("--max-sample-index", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.max_sample_index <= 0:
        raise ValueError("max_sample_index must be positive")

    prompts = _load_prompts(args.prompts)
    method_paths = _parse_methods(args.method)
    candidates: dict[str, list[dict]] = {}
    source_hashes: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    for method, path in method_paths.items():
        candidates[method], source_hashes[method] = _load_score_dir(
            path, max_sample_index=args.max_sample_index
        )
        source_counts[method] = len(candidates[method])
    entries = build_blinded_calibration_sample(
        candidates_by_method=candidates,
        prompts=prompts,
        samples_per_method=args.samples_per_method,
        repeat_fraction=args.repeat_fraction,
        seed=args.seed,
    )
    payload = {
        "schema_version": 1,
        "protocol_hash": independent_judge_protocol_hash(),
        "sampling": {
            "design": "paired-unique-prompts-pooled-qwen-core-rank-quartiles-v2",
            "samples_per_method": args.samples_per_method,
            "repeat_fraction": args.repeat_fraction,
            "max_sample_index_exclusive": args.max_sample_index,
            "seed": args.seed,
            "methods": sorted(method_paths),
        },
        "sources": {
            "prompt_path": str(args.prompts.resolve()),
            "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "score_directories": {
                method: str(path.resolve()) for method, path in sorted(method_paths.items())
            },
            "score_directory_fingerprints": source_hashes,
            "candidate_counts": source_counts,
        },
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "methods": sorted(method_paths),
                "originals": sum(entry.repeat_of is None for entry in entries),
                "repeats": sum(entry.repeat_of is not None for entry in entries),
                "total": len(entries),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
