"""Strictly merge independently embedded teacher-clustering partitions."""

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from novelty_distill.data.teacher_views import TeacherGeneration

_PARTITION_FIELDS = {
    "global_num_prompts",
    "num_prompts",
    "num_records",
    "num_shards",
    "prompt_diagnostics",
    "score_files",
    "shard_index",
}


def merge_cluster_shards(
    shard_paths: Sequence[Path],
    *,
    output: Path,
    expected_prompts: int,
    expected_num_shards: int,
) -> dict[str, Any]:
    """Merge a complete partition set, rejecting gaps, drift, and duplicate prompts."""

    if expected_prompts <= 0 or expected_num_shards <= 0:
        raise ValueError("expected prompt and shard counts must be positive")
    if not shard_paths:
        raise ValueError("no teacher cluster partitions were provided")

    common: dict[str, Any] | None = None
    indexes: set[int] = set()
    prompt_records: dict[str, list[TeacherGeneration]] = {}
    prompt_files: dict[str, str] = {}
    score_names: set[str] = set()
    diagnostics: dict[str, object] = {}
    total_records = 0
    for path in shard_paths:
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 2:
            raise ValueError(f"unsupported cluster partition metadata in {metadata_path}")
        if metadata.get("num_shards") != expected_num_shards:
            raise ValueError(f"cluster partition count changed in {metadata_path}")
        if metadata.get("global_num_prompts") != expected_prompts:
            raise ValueError(f"global prompt count changed in {metadata_path}")
        index = int(metadata.get("shard_index", -1))
        if index in indexes:
            raise ValueError(f"duplicate cluster partition index {index}")
        indexes.add(index)
        current_common = {
            key: value for key, value in metadata.items() if key not in _PARTITION_FIELDS
        }
        if common is None:
            common = current_common
        elif current_common != common:
            raise ValueError(f"cluster controls changed in {metadata_path}")

        file_entries = metadata.get("score_files")
        part_diagnostics = metadata.get("prompt_diagnostics")
        if not isinstance(file_entries, list) or not isinstance(part_diagnostics, dict):
            raise ValueError(f"invalid cluster partition metadata in {metadata_path}")
        records: list[TeacherGeneration] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(TeacherGeneration.model_validate_json(line))
        if len(records) != metadata.get("num_records"):
            raise ValueError(f"cluster record count mismatch in {path}")
        grouped: dict[str, list[TeacherGeneration]] = {}
        for record in records:
            grouped.setdefault(record.prompt_id, []).append(record)
        if len(grouped) != metadata.get("num_prompts"):
            raise ValueError(f"cluster prompt count mismatch in {path}")
        if set(grouped) != set(part_diagnostics):
            raise ValueError(f"cluster diagnostics differ from records in {path}")
        for entry in file_entries:
            if not isinstance(entry, dict):
                raise ValueError(f"invalid score file entry in {metadata_path}")
            prompt_id = entry.get("prompt_id")
            name = entry.get("name")
            if prompt_id not in grouped or not isinstance(name, str) or not name:
                raise ValueError(f"invalid score file mapping in {metadata_path}")
            if prompt_id in prompt_records or name in score_names:
                raise ValueError(f"duplicate clustered prompt or score file {prompt_id!r}")
            ordered = sorted(grouped[prompt_id], key=lambda record: record.sample_index)
            if [record.sample_index for record in ordered] != list(range(len(ordered))):
                raise ValueError(f"non-contiguous samples for clustered prompt {prompt_id!r}")
            prompt_records[prompt_id] = ordered
            prompt_files[prompt_id] = name
            score_names.add(name)
            diagnostics[prompt_id] = part_diagnostics[prompt_id]
            total_records += len(ordered)

    expected_indexes = set(range(expected_num_shards))
    if indexes != expected_indexes:
        raise ValueError(
            f"cluster partition indexes differ: expected {sorted(expected_indexes)}, "
            f"found {sorted(indexes)}"
        )
    if len(prompt_records) != expected_prompts:
        raise ValueError(
            f"clustered prompt count differs: expected {expected_prompts}, "
            f"found {len(prompt_records)}"
        )
    assert common is not None
    ordered_prompts = sorted(prompt_records, key=lambda prompt_id: prompt_files[prompt_id])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for prompt_id in ordered_prompts:
                for record in prompt_records[prompt_id]:
                    handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)

    merged = {
        **common,
        "num_records": total_records,
        "num_prompts": len(prompt_records),
        "prompt_diagnostics": diagnostics,
        "score_files": [
            {"name": prompt_files[prompt_id], "prompt_id": prompt_id}
            for prompt_id in ordered_prompts
        ],
        "merged_shards": expected_num_shards,
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return merged
