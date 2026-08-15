"""Materialize normalized semantic banks from content-addressed embedding caches."""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import yaml

from novelty_distill.data.teacher_views import TeacherGeneration
from novelty_distill.evaluation.embeddings import EmbeddingCache


def write_semantic_bank(
    *,
    clustered_path: Path,
    output_path: Path,
    annotation_config: Path,
    embedding_cache_dir: Path,
    role: Literal["teacher", "student"],
) -> int:
    """Write coverage-selector records without re-embedding clustered text."""

    annotation = yaml.safe_load(annotation_config.read_text(encoding="utf-8"))
    cache = EmbeddingCache(
        embedding_cache_dir,
        model_id=str(annotation["embedding_model"]),
        revision=str(annotation["embedding_revision"]),
        max_length=int(annotation["embedding_max_length"]),
        batch_size=int(annotation["embedding_batch_size"]),
    )
    instruction = str(annotation["embedding_instruction"])
    records = tuple(
        TeacherGeneration.model_validate_json(line)
        for line in clustered_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError(f"clustered generation file is empty: {clustered_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for record in records:
                cache_key = f"Instruct: {instruction}\nQuery: {record.text}"
                embedding = cache.get(cache_key)
                if embedding is None:
                    raise ValueError(
                        f"missing instructed embedding cache entry for "
                        f"{record.prompt_id}/{record.sample_index}"
                    )
                payload: dict[str, object] = {
                    "prompt_id": record.prompt_id,
                    "sample_index": record.sample_index,
                    "embedding": embedding,
                }
                if role == "teacher":
                    payload.update(
                        text=record.text,
                        quality_score=record.quality_score,
                    )
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return len(records)
