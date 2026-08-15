import json
from pathlib import Path

import numpy as np
import yaml

from novelty_distill.evaluation.embeddings import EmbeddingCache
from novelty_distill.evaluation.semantic_bank import write_semantic_bank


def test_semantic_bank_reuses_frozen_instructed_embedding_cache(tmp_path: Path) -> None:
    clustered = tmp_path / "clustered.jsonl"
    clustered.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "sample_index": 0,
                "text": "A mechanism and test.",
                "quality_score": 0.8,
                "cluster_id": "cluster-000",
            }
        )
        + "\n"
    )
    config = tmp_path / "annotation.yaml"
    controls = {
        "embedding_model": "embedding/model",
        "embedding_revision": "a" * 40,
        "embedding_instruction": "Represent the idea.",
        "embedding_batch_size": 8,
        "embedding_max_length": 2048,
    }
    config.write_text(yaml.safe_dump(controls))
    cache_root = tmp_path / "cache"
    cache = EmbeddingCache(
        cache_root,
        model_id=controls["embedding_model"],
        revision=controls["embedding_revision"],
        max_length=controls["embedding_max_length"],
        batch_size=controls["embedding_batch_size"],
    )
    cache.put(
        "Instruct: Represent the idea.\nQuery: A mechanism and test.",
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    teacher = tmp_path / "teacher.jsonl"
    student = tmp_path / "student.jsonl"
    write_semantic_bank(
        clustered_path=clustered,
        output_path=teacher,
        annotation_config=config,
        embedding_cache_dir=cache_root,
        role="teacher",
    )
    write_semantic_bank(
        clustered_path=clustered,
        output_path=student,
        annotation_config=config,
        embedding_cache_dir=cache_root,
        role="student",
    )

    teacher_record = json.loads(teacher.read_text())
    student_record = json.loads(student.read_text())
    assert teacher_record == {
        "embedding": [1.0, 0.0],
        "prompt_id": "p1",
        "quality_score": 0.8,
        "sample_index": 0,
        "text": "A mechanism and test.",
    }
    assert student_record == {
        "embedding": [1.0, 0.0],
        "prompt_id": "p1",
        "sample_index": 0,
    }
