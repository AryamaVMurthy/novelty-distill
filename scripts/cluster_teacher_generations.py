#!/usr/bin/env python3
"""Embed scored teacher samples and assign deterministic semantic clusters."""

import argparse
import json
from pathlib import Path

import yaml

from novelty_distill.data.teacher_views import TeacherGeneration
from novelty_distill.evaluation.teacher_annotation import (
    cluster_cosine_embeddings,
    cosine_embedding_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    return parser.parse_args()


def _embed(
    texts: list[str],
    *,
    model_id: str,
    revision: str,
    max_length: int,
    batch_size: int,
) -> list[list[float]]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    embeddings: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)
            hidden = model(**batch).last_hidden_state
            pooled = hidden[:, -1]
            normalized = functional.normalize(pooled.float(), p=2, dim=1)
            embeddings.extend(normalized.cpu().tolist())
    return embeddings


def main() -> None:
    args = parse_args()
    annotation = yaml.safe_load(args.annotation_config.read_text(encoding="utf-8"))
    score_paths = sorted(args.score_dir.glob("*.json"))
    if not score_paths:
        raise ValueError(f"no score shards found in {args.score_dir}")

    output_records: list[TeacherGeneration] = []
    prompt_diagnostics: dict[str, object] = {}
    for score_path in score_paths:
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if len(records) != 8:
            raise ValueError(f"score shard {score_path} must contain exactly eight records")
        texts = [record["text"] for record in records]
        instruction = str(annotation["embedding_instruction"])
        embedding_inputs = texts + [
            f"Instruct: {instruction}\nQuery: {text}" for text in texts
        ]
        all_embeddings = _embed(
            embedding_inputs,
            model_id=annotation["embedding_model"],
            revision=annotation["embedding_revision"],
            max_length=annotation["embedding_max_length"],
            batch_size=int(annotation["embedding_batch_size"]),
        )
        raw_embeddings = all_embeddings[: len(texts)]
        instructed_embeddings = all_embeddings[len(texts) :]
        labels = cluster_cosine_embeddings(
            instructed_embeddings, threshold=float(annotation["cosine_threshold"])
        )
        prompt_id = str(records[0]["prompt_id"])
        if any(str(record["prompt_id"]) != prompt_id for record in records):
            raise ValueError(f"score shard {score_path} contains multiple prompt ids")
        diagnostic_thresholds = tuple(
            float(value) for value in annotation["cosine_thresholds"]
        )
        prompt_diagnostics[prompt_id] = {
            "primary_embedding": "instructed",
            "judge_dimensions": [record["dimensions"] for record in records],
            "raw": cosine_embedding_diagnostics(
                raw_embeddings, thresholds=diagnostic_thresholds
            ),
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(record.model_dump_json() + "\n")
    temporary.replace(args.output)
    metadata = {
        "schema_version": 1,
        "num_records": len(output_records),
        "num_prompts": len(score_paths),
        "embedding_model": annotation["embedding_model"],
        "embedding_revision": annotation["embedding_revision"],
        "embedding_instruction": annotation["embedding_instruction"],
        "embedding_batch_size": annotation["embedding_batch_size"],
        "cosine_threshold": annotation["cosine_threshold"],
        "cosine_thresholds": annotation["cosine_thresholds"],
        "prompt_diagnostics": prompt_diagnostics,
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
