"""Validation shared by quality-scoring workers and their cheap resume preflight."""

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.teacher_annotation import JudgeSpec, QualityDimensions
from novelty_distill.generation.sglang import GenerationSpec, load_prompt_shard


def validate_score_shard(
    path: Path,
    *,
    prompt_id: str,
    text_hashes: list[str],
    judge: JudgeSpec,
) -> bool:
    """Return false for a missing score shard and reject incompatible existing data."""

    if not path.exists():
        return False
    try:
        existing = load_score_shard(path, samples_per_prompt=len(text_hashes))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"stale or incompatible score shard {path}") from error
    if (
        existing.get("prompt_id") != prompt_id
        or existing.get("text_hashes") != text_hashes
        or existing.get("judge") != judge.model_dump(mode="json")
    ):
        raise ValueError(f"stale or incompatible score shard {path}")
    return True


def load_score_shard(path: Path, *, samples_per_prompt: int) -> dict[str, Any]:
    """Load one score shard and verify its records, hashes, and normalized scores."""

    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError(f"unsupported score shard schema in {path}")
    prompt_id = payload.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError(f"invalid score prompt ID in {path}")
    JudgeSpec.model_validate(payload.get("judge"))
    text_hashes = payload.get("text_hashes")
    records = payload.get("records")
    if (
        not isinstance(text_hashes, list)
        or len(text_hashes) != samples_per_prompt
        or any(not isinstance(value, str) or len(value) != 64 for value in text_hashes)
    ):
        raise ValueError(f"invalid score text hashes in {path}")
    if not isinstance(records, list) or len(records) != samples_per_prompt:
        raise ValueError(f"score shard {path} must contain {samples_per_prompt} records")
    ordered = sorted(records, key=lambda record: int(record.get("sample_index", -1)))
    if [int(record.get("sample_index", -1)) for record in ordered] != list(
        range(samples_per_prompt)
    ):
        raise ValueError(f"non-contiguous score sample indices in {path}")
    request_ids: set[str] = set()
    for index, record in enumerate(ordered):
        if not isinstance(record, Mapping) or record.get("prompt_id") != prompt_id:
            raise ValueError(f"inconsistent score record prompt in {path}")
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty score record text in {path}")
        if hashlib.sha256(text.encode()).hexdigest() != text_hashes[index]:
            raise ValueError(f"score record text hash mismatch in {path}")
        dimensions = QualityDimensions.model_validate(record.get("dimensions"))
        raw_quality = record.get("quality_score")
        if isinstance(raw_quality, bool):
            raise ValueError(f"invalid quality score in {path}")
        quality = float(raw_quality)
        expected_quality = (sum(dimensions.model_dump().values()) / 5 - 1) / 4
        if not math.isfinite(quality) or not math.isclose(quality, expected_quality, abs_tol=1e-12):
            raise ValueError(f"quality score does not match dimensions in {path}")
        finish_reason = record.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ValueError(f"invalid finish reason in {path}")
        completion_tokens = record.get("completion_tokens")
        if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int):
            raise ValueError(f"invalid completion token count in {path}")
        if completion_tokens < 0:
            raise ValueError(f"negative completion token count in {path}")
        request_id = record.get("request_id")
        model = record.get("model")
        if not isinstance(request_id, str) or not request_id or request_id in request_ids:
            raise ValueError(f"invalid or duplicate judge request ID in {path}")
        if not isinstance(model, str) or not model:
            raise ValueError(f"invalid judge model in {path}")
        request_ids.add(request_id)
    payload["records"] = ordered
    return payload


def score_run_status(
    *,
    generation_dir: Path,
    output_dir: Path,
    generation_spec: GenerationSpec,
    judge: JudgeSpec,
) -> tuple[int, int]:
    """Return total and pending prompt shards after strictly validating completed scores."""

    total = 0
    pending = 0
    for generation_path in sorted(generation_dir.glob("*.json")):
        records = load_prompt_shard(generation_path, generation_spec)
        prompt_id = records[0].prompt_id
        text_hashes = [hashlib.sha256(record.text.encode()).hexdigest() for record in records]
        if not validate_score_shard(
            output_dir / generation_path.name,
            prompt_id=prompt_id,
            text_hashes=text_hashes,
            judge=judge,
        ):
            pending += 1
        total += 1
    if total == 0:
        raise ValueError(f"no generation shards found in {generation_dir}")
    return total, pending
