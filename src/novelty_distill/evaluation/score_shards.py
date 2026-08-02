"""Validation shared by quality-scoring workers and their cheap resume preflight."""

import hashlib
import json
from pathlib import Path

from novelty_distill.evaluation.teacher_annotation import JudgeSpec
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
    existing = json.loads(path.read_text(encoding="utf-8"))
    if (
        existing.get("schema_version") != 2
        or existing.get("prompt_id") != prompt_id
        or existing.get("text_hashes") != text_hashes
        or existing.get("judge") != judge.model_dump(mode="json")
    ):
        raise ValueError(f"stale or incompatible score shard {path}")
    return True


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
