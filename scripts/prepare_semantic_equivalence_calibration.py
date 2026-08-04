#!/usr/bin/env python3
"""Prepare source-blinded human labels for the embedding equivalence boundary."""

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.embeddings import EmbeddingCache
from novelty_distill.evaluation.semantic_calibration import (
    build_semantic_calibration_sample,
    semantic_calibration_protocol_hash,
)

DEFAULT_BINS = (
    (0.74, 0.84),
    (0.84, 0.88),
    (0.88, 0.90),
    (0.90, 0.92),
    (0.92, 0.94),
    (0.94, 0.96),
    (0.96, 0.98),
    (0.98, 1.001),
)


def _atomic_json(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=indent, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
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
            if line.strip():
                row = json.loads(line)
                prompt_id = str(row["id"])
                if prompt_id in prompts:
                    raise ValueError(f"duplicate prompt {prompt_id}")
                prompts[prompt_id] = str(row["student_prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-bin", type=int, default=32)
    parser.add_argument("--repeat-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts)
    target_bytes = args.teacher_targets.read_bytes()
    targets_payload = json.loads(target_bytes)
    targets = targets_payload.get("targets")
    if targets_payload.get("schema_version") != 1 or not isinstance(targets, dict):
        raise ValueError("teacher targets must be a schema-v1 target mapping")
    annotation_bytes = args.annotation_config.read_bytes()
    annotation = yaml.safe_load(annotation_bytes)
    cache = EmbeddingCache(
        args.embedding_cache_dir,
        model_id=annotation["embedding_model"],
        revision=annotation["embedding_revision"],
        max_length=int(annotation["embedding_max_length"]),
        batch_size=int(annotation["embedding_batch_size"]),
    )
    instruction = str(annotation["embedding_instruction"])
    pairs: list[dict[str, Any]] = []
    for prompt_id, views in sorted(targets.items()):
        if prompt_id not in prompts:
            raise ValueError(f"teacher target prompt {prompt_id} is absent from prompts")
        texts = views.get("all8") if isinstance(views, dict) else None
        if not isinstance(texts, list) or len(texts) != 8 or len(set(texts)) != 8:
            raise ValueError(f"teacher target {prompt_id} does not contain eight distinct texts")
        vectors = []
        for text in texts:
            cache_input = f"Instruct: {instruction}\nQuery: {text}"
            vector = cache.get_array(cache_input)
            if vector is None:
                raise ValueError(f"embedding cache miss for teacher target {prompt_id}")
            vectors.append(vector)
        for left_index, right_index in itertools.combinations(range(8), 2):
            pairs.append(
                {
                    "prompt_id": prompt_id,
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_text": texts[left_index],
                    "right_text": texts[right_index],
                    "similarity": float(vectors[left_index] @ vectors[right_index]),
                }
            )

    sample = build_semantic_calibration_sample(
        pairs=pairs,
        prompts=prompts,
        similarity_bins=DEFAULT_BINS,
        pairs_per_bin=args.pairs_per_bin,
        repeat_fraction=args.repeat_fraction,
        seed=args.seed,
    )
    protocol_hash = semantic_calibration_protocol_hash()
    public_payload = {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "labels": ["equivalent", "not_equivalent", "uncertain"],
        "entries": [entry.model_dump(mode="json") for entry in sample.public_entries],
    }
    private_payload = {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "sampling": {
            "design": "clean-teacher-pairs-global-unique-prompts-stratified-v1",
            "similarity_bins": DEFAULT_BINS,
            "pairs_per_bin": args.pairs_per_bin,
            "repeat_fraction": args.repeat_fraction,
            "seed": args.seed,
        },
        "sources": {
            "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "teacher_targets_sha256": hashlib.sha256(target_bytes).hexdigest(),
            "annotation_config_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
            "embedding_cache_fingerprint": cache.fingerprint,
            "candidate_pairs": len(pairs),
        },
        "entries": [entry.model_dump(mode="json") for entry in sample.private_entries],
    }
    public_path = args.output_dir / "semantic-equivalence-public.json"
    private_path = args.output_dir / "semantic-equivalence-private-key.json"
    _atomic_json(public_path, public_payload)
    _atomic_json(private_path, private_payload)
    template = [
        {"blind_id": entry.blind_id, "label": "", "confidence": None, "rationale": ""}
        for entry in sample.public_entries
    ]
    _atomic_jsonl(args.output_dir / "labels-rater-one.jsonl", template)
    _atomic_jsonl(args.output_dir / "labels-rater-two.jsonl", template)
    manifest = {
        "schema_version": 1,
        "status": "prepared_awaiting_two_human_raters",
        "protocol_hash": protocol_hash,
        "public_packet_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "private_key_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
        "originals": sum(entry.repeat_of is None for entry in sample.private_entries),
        "hidden_repeats": sum(entry.repeat_of is not None for entry in sample.private_entries),
        "total_rows_per_rater": len(sample.private_entries),
        "claim_boundary": "Calibrates semantic equivalence only, not scientific novelty.",
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
