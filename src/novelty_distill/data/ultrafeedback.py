"""Deterministic pilot views over the pinned official UltraFeedback records."""

import hashlib
import itertools
import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from novelty_distill.data.tomato import CanonicalExample, PrivilegedContext

ULTRAFEEDBACK_DATASET_ID = "openbmb/UltraFeedback"
ULTRAFEEDBACK_REVISION = "40b436560ca83a8dba36114c22ab3c66e43f6d5e"


class UltraFeedbackCompletion(BaseModel):
    """The stable fields needed from one of four official candidate completions."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str = Field(min_length=1)
    response: str = Field(min_length=1)
    overall_score: float
    fine_grained_score: float = Field(validation_alias="fine-grained_score")


def ultrafeedback_record_id(raw: Mapping[str, Any]) -> str:
    """Hash content fields that distinguish duplicate official instructions."""

    raw_completions = raw.get("completions")
    if not isinstance(raw_completions, list):
        raise ValueError("UltraFeedback completions must be a list")
    identity = {
        "source": str(raw.get("source", "")).strip(),
        "instruction": str(raw.get("instruction", "")).strip(),
        "completions": sorted(
            (
                str(completion.get("model", "")),
                str(completion.get("response", "")),
            )
            for completion in raw_completions
            if isinstance(completion, Mapping)
        ),
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return "ultrafeedback-" + hashlib.sha256(encoded).hexdigest()


def select_unique_content_indices(
    identities: tuple[str, ...], *, size: int, seed: int
) -> tuple[int, ...]:
    """Select a deterministic subset after dropping exact-content duplicates."""

    if size <= 0:
        raise ValueError("pilot size must be positive")
    first_index_by_identity: dict[str, int] = {}
    for index, identity in enumerate(identities):
        if not identity:
            raise ValueError("content identities must be non-empty")
        first_index_by_identity.setdefault(identity, index)
    if size > len(first_index_by_identity):
        raise ValueError(
            f"requested {size} rows from {len(first_index_by_identity)} unique records"
        )
    return tuple(
        sorted(
            first_index_by_identity.values(),
            key=lambda index: (
                hashlib.sha256(f"{seed}\0{identities[index]}".encode()).digest(),
                identities[index],
            ),
        )[:size]
    )


def prepare_ultrafeedback_record(
    raw: Mapping[str, Any], *, seed: int
) -> tuple[CanonicalExample, dict[str, list[str]]]:
    """Convert one official record and derive random, best, all, and diverse views."""

    instruction = str(raw.get("instruction", "")).strip()
    source = str(raw.get("source", "")).strip()
    if not instruction or not source:
        raise ValueError("UltraFeedback instruction and source must be non-empty")
    raw_completions = raw.get("completions")
    if not isinstance(raw_completions, list) or len(raw_completions) != 4:
        raise ValueError("UltraFeedback record must contain exactly four completions")
    completions = tuple(
        sorted(
            (UltraFeedbackCompletion.model_validate(value) for value in raw_completions),
            key=lambda completion: completion.model,
        )
    )
    models = tuple(str(value) for value in raw.get("models", ()))
    if len(set(models)) != 4 or set(models) != {completion.model for completion in completions}:
        raise ValueError("UltraFeedback model and completion identities must align")

    record_id = ultrafeedback_record_id(raw)
    best = min(
        completions,
        key=lambda completion: (
            -completion.overall_score,
            -completion.fine_grained_score,
            completion.model,
        ),
    )
    random_digest = hashlib.sha256(f"{seed}\0{record_id}".encode()).digest()
    random_completion = completions[int.from_bytes(random_digest[:8], "big") % len(completions)]
    diverse_pair = min(
        itertools.combinations(completions, 2),
        key=lambda pair: (
            -_jaccard_distance(pair[0].response, pair[1].response),
            -(pair[0].overall_score + pair[1].overall_score),
            pair[0].model,
            pair[1].model,
        ),
    )
    diverse_pair = tuple(
        sorted(
            diverse_pair,
            key=lambda completion: (-completion.overall_score, completion.model),
        )
    )
    all_texts = [completion.response for completion in completions]
    targets = {
        "random1": [random_completion.response],
        "best1": [best.response],
        "mode1": [best.response],
        "diverse2": [completion.response for completion in diverse_pair],
        "diverse4": all_texts,
        "all4": all_texts,
    }
    example = CanonicalExample(
        id=record_id,
        student_prompt=instruction,
        privileged_context=PrivilegedContext(
            historical_hypothesis=best.response,
            inspirations=(),
        ),
        human_target=best.response,
        source_ids=(source,),
        split="pilot",
        task="open",
        prompt_hash=hashlib.sha256(instruction.encode()).hexdigest(),
        dataset_revision=ULTRAFEEDBACK_REVISION,
    )
    return example, targets


def _jaccard_distance(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", left.casefold()))
    right_tokens = set(re.findall(r"\w+", right.casefold()))
    union = left_tokens | right_tokens
    if not union:
        return 0
    return 1 - len(left_tokens & right_tokens) / len(union)
