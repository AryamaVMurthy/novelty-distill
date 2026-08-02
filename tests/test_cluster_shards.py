import json
from pathlib import Path

import pytest

from novelty_distill.evaluation.cluster_shards import merge_cluster_shards


def _write_part(
    root: Path, *, shard_index: int, prompt_id: str, score_file: str
) -> Path:
    path = root / f"part-{shard_index:05d}-of-00002.jsonl"
    record = {
        "prompt_id": prompt_id,
        "sample_index": 0,
        "text": f"response-{prompt_id}",
        "quality_score": 0.75,
        "cluster_id": "0",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": 2,
        "git_commit": "a" * 40,
        "num_records": 1,
        "num_prompts": 1,
        "embedding_model": "embedding-model",
        "embedding_revision": "b" * 40,
        "embedding_instruction": "cluster ideas",
        "embedding_batch_size": 8,
        "clustering_linkage": "complete",
        "cosine_threshold": 0.94,
        "cosine_thresholds": [0.9, 0.94],
        "judge": {"model": "judge", "revision": "c" * 40},
        "prompt_diagnostics": {prompt_id: {"primary_embedding": "instructed"}},
        "score_files": [{"name": score_file, "prompt_id": prompt_id}],
        "shard_index": shard_index,
        "num_shards": 2,
        "global_num_prompts": 2,
    }
    path.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return path


def test_merge_cluster_shards_restores_global_score_file_order(tmp_path: Path) -> None:
    second = _write_part(tmp_path, shard_index=0, prompt_id="p2", score_file="b.json")
    first = _write_part(tmp_path, shard_index=1, prompt_id="p1", score_file="a.json")
    output = tmp_path / "merged.jsonl"

    metadata = merge_cluster_shards(
        (second, first), output=output, expected_prompts=2, expected_num_shards=2
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["prompt_id"] for record in records] == ["p1", "p2"]
    assert metadata["num_prompts"] == 2
    assert metadata["merged_shards"] == 2
    assert set(metadata["prompt_diagnostics"]) == {"p1", "p2"}


def test_merge_cluster_shards_rejects_missing_partition(tmp_path: Path) -> None:
    only = _write_part(tmp_path, shard_index=0, prompt_id="p1", score_file="a.json")

    with pytest.raises(ValueError, match="partition indexes"):
        merge_cluster_shards(
            (only,),
            output=tmp_path / "merged.jsonl",
            expected_prompts=2,
            expected_num_shards=2,
        )
