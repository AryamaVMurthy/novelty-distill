#!/usr/bin/env python3
"""Embed scored teacher samples and assign deterministic semantic clusters."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.data.teacher_views import TeacherGeneration
from novelty_distill.evaluation.embeddings import embed_texts
from novelty_distill.evaluation.score_shards import load_score_shard, shard_score_paths
from novelty_distill.evaluation.teacher_annotation import (
    cluster_cosine_embeddings,
    cosine_embedding_diagnostics,
)
from novelty_distill.provenance import repository_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    git_commit = repository_commit(Path(__file__).resolve().parents[1])
    annotation = yaml.safe_load(args.annotation_config.read_text(encoding="utf-8"))
    if annotation.get("clustering_linkage") != "complete":
        raise ValueError("teacher clustering requires deterministic complete linkage")
    all_score_paths = sorted(args.score_dir.glob("*.json"))
    if not all_score_paths:
        raise ValueError(f"no score shards found in {args.score_dir}")
    score_paths = shard_score_paths(
        all_score_paths, num_shards=args.num_shards, shard_index=args.shard_index
    )
    if not score_paths:
        raise ValueError(
            f"cluster partition {args.shard_index} has no inputs from {args.score_dir}"
        )

    output_records: list[TeacherGeneration] = []
    prompt_diagnostics: dict[str, object] = {}
    prompt_batches: list[tuple[Path, list[dict[str, Any]]]] = []
    embedding_inputs: list[str] = []
    judge_payload: dict[str, Any] | None = None
    prompt_ids: set[str] = set()
    instruction = str(annotation["embedding_instruction"])
    for score_path in score_paths:
        payload = load_score_shard(score_path, samples_per_prompt=8)
        records = payload["records"]
        current_judge = payload["judge"]
        if judge_payload is None:
            judge_payload = current_judge
        elif current_judge != judge_payload:
            raise ValueError(f"score shard {score_path} changed the judge specification")
        prompt_id = str(payload["prompt_id"])
        if prompt_id in prompt_ids:
            raise ValueError(f"duplicate score prompt ID {prompt_id!r}")
        prompt_ids.add(prompt_id)
        texts = [record["text"] for record in records]
        prompt_batches.append((score_path, records))
        embedding_inputs.extend(texts)
        embedding_inputs.extend(f"Instruct: {instruction}\nQuery: {text}" for text in texts)

    all_embeddings = embed_texts(
        embedding_inputs,
        model_id=annotation["embedding_model"],
        revision=annotation["embedding_revision"],
        max_length=annotation["embedding_max_length"],
        batch_size=int(annotation["embedding_batch_size"]),
        cache_dir=args.embedding_cache_dir,
    )
    embedding_cursor = 0
    for score_path, records in prompt_batches:
        sample_count = len(records)
        raw_embeddings = all_embeddings[embedding_cursor : embedding_cursor + sample_count]
        instructed_embeddings = all_embeddings[
            embedding_cursor + sample_count : embedding_cursor + 2 * sample_count
        ]
        embedding_cursor += 2 * sample_count
        labels = cluster_cosine_embeddings(
            instructed_embeddings, threshold=float(annotation["cosine_threshold"])
        )
        prompt_id = str(records[0]["prompt_id"])
        if any(str(record["prompt_id"]) != prompt_id for record in records):
            raise ValueError(f"score shard {score_path} contains multiple prompt ids")
        diagnostic_thresholds = tuple(float(value) for value in annotation["cosine_thresholds"])
        prompt_diagnostics[prompt_id] = {
            "primary_embedding": "instructed",
            "judge_dimensions": [record["dimensions"] for record in records],
            "raw": cosine_embedding_diagnostics(raw_embeddings, thresholds=diagnostic_thresholds),
            "instructed": cosine_embedding_diagnostics(
                instructed_embeddings, thresholds=diagnostic_thresholds
            ),
        }
        for record, cluster_id in zip(records, labels, strict=True):
            output_records.append(
                TeacherGeneration(
                    prompt_id=record["prompt_id"],
                    sample_index=record["sample_index"],
                    text=record["text"],
                    quality_score=record["quality_score"],
                    cluster_id=cluster_id,
                )
            )

    if embedding_cursor != len(all_embeddings):
        raise AssertionError("embedding batch cursor did not consume the full corpus")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(record.model_dump_json() + "\n")
    temporary.replace(args.output)
    metadata = {
        "schema_version": 2,
        "git_commit": git_commit,
        "num_records": len(output_records),
        "num_prompts": len(score_paths),
        "global_num_prompts": len(all_score_paths),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "embedding_model": annotation["embedding_model"],
        "embedding_revision": annotation["embedding_revision"],
        "embedding_instruction": annotation["embedding_instruction"],
        "embedding_batch_size": annotation["embedding_batch_size"],
        "clustering_linkage": annotation["clustering_linkage"],
        "cosine_threshold": annotation["cosine_threshold"],
        "cosine_thresholds": annotation["cosine_thresholds"],
        "judge": judge_payload,
        "prompt_diagnostics": prompt_diagnostics,
        "score_files": [
            {"name": path.name, "prompt_id": str(records[0]["prompt_id"])}
            for path, records in prompt_batches
        ],
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
