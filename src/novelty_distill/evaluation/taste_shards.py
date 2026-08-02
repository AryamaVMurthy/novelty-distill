"""Fail-closed validation for resumable research-taste annotation shards."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.research_taste import (
    JudgedResearchTaste,
    ResearchTasteSpec,
)
from novelty_distill.generation.sglang import GenerationSpec, load_prompt_shard


def validate_research_taste_shard(
    path: Path,
    *,
    prompt_id: str,
    text_hashes: list[str],
    annotator: ResearchTasteSpec,
) -> bool:
    """Return false when absent and reject an incompatible existing shard."""

    if not path.exists():
        return False
    try:
        existing = load_research_taste_shard(path, samples_per_prompt=len(text_hashes))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"stale or incompatible research-taste shard {path}") from error
    if (
        existing.get("prompt_id") != prompt_id
        or existing.get("text_hashes") != text_hashes
        or existing.get("annotator") != annotator.model_dump(mode="json")
    ):
        raise ValueError(f"stale or incompatible research-taste shard {path}")
    return True


def load_research_taste_shard(
    path: Path, *, samples_per_prompt: int
) -> dict[str, Any]:
    """Load one annotation shard and verify identity, hashes, and strict labels."""

    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported research-taste shard schema in {path}")
    prompt_id = payload.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError(f"invalid research-taste prompt ID in {path}")
    ResearchTasteSpec.model_validate(payload.get("annotator"))
    text_hashes = payload.get("text_hashes")
    records = payload.get("records")
    if (
        not isinstance(text_hashes, list)
        or len(text_hashes) != samples_per_prompt
        or any(not isinstance(value, str) or len(value) != 64 for value in text_hashes)
    ):
        raise ValueError(f"invalid research-taste text hashes in {path}")
    if not isinstance(records, list) or len(records) != samples_per_prompt:
        raise ValueError(
            f"research-taste shard {path} must contain {samples_per_prompt} records"
        )
    ordered = sorted(records, key=lambda record: int(record.get("sample_index", -1)))
    if [int(record.get("sample_index", -1)) for record in ordered] != list(
        range(samples_per_prompt)
    ):
        raise ValueError(f"non-contiguous research-taste sample indices in {path}")
    request_ids: set[str] = set()
    for index, record in enumerate(ordered):
        if not isinstance(record, Mapping) or record.get("prompt_id") != prompt_id:
            raise ValueError(f"inconsistent research-taste record prompt in {path}")
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty research-taste record text in {path}")
        if hashlib.sha256(text.encode()).hexdigest() != text_hashes[index]:
            raise ValueError(f"research-taste record text hash mismatch in {path}")
        judged = JudgedResearchTaste.model_validate(
            {
                key: value
                for key, value in record.items()
                if key not in {"prompt_id", "sample_index", "text"}
            }
        )
        if judged.request_id in request_ids:
            raise ValueError(f"duplicate annotator request ID in {path}")
        request_ids.add(judged.request_id)
    payload["records"] = ordered
    return payload


def research_taste_run_status(
    *,
    generation_dir: Path,
    output_dir: Path,
    generation_spec: GenerationSpec,
    annotator: ResearchTasteSpec,
) -> tuple[int, int]:
    """Return total and pending prompt shards after validating completed work."""

    total = 0
    pending = 0
    for generation_path in sorted(generation_dir.glob("*.json")):
        records = load_prompt_shard(generation_path, generation_spec)
        prompt_id = records[0].prompt_id
        text_hashes = [hashlib.sha256(record.text.encode()).hexdigest() for record in records]
        if not validate_research_taste_shard(
            output_dir / generation_path.name,
            prompt_id=prompt_id,
            text_hashes=text_hashes,
            annotator=annotator,
        ):
            pending += 1
        total += 1
    if total == 0:
        raise ValueError(f"no generation shards found in {generation_dir}")
    return total, pending
