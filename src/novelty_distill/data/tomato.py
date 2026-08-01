"""Adapter for the official ZonglinY/TOMATO-Star schema."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

TOMATO_DATASET_ID = "ZonglinY/TOMATO-Star"
TOMATO_REVISION = "fcd201d92758a642465a7653b8055f3a04d5f439"


class PrivilegedContext(BaseModel):
    """Information visible only to privileged-teacher baselines."""

    model_config = ConfigDict(frozen=True)

    historical_hypothesis: str
    inspirations: tuple[str, ...]


class CanonicalExample(BaseModel):
    """Shared example consumed by every baseline family."""

    model_config = ConfigDict(frozen=True)

    id: str
    student_prompt: str
    privileged_context: PrivilegedContext
    human_target: str
    source_ids: tuple[str, ...]
    split: str
    task: Literal["open", "composition"]
    prompt_hash: str
    dataset_revision: str = TOMATO_REVISION


def prepare_tomato_record(
    raw: Mapping[str, Any], *, split: str, task: Literal["open", "composition"]
) -> CanonicalExample:
    """Convert one official TOMATO-Star record without exposing identity fields."""

    question = str(raw["research_question"]).strip()
    background = str(raw["background_survey"]).strip()
    target = str(raw["fine_grained_hypothesis"]).strip()
    source_id = str(raw["source_id"]).strip()
    raw_inspirations = raw.get("inspiration", ())
    if isinstance(raw_inspirations, str):
        try:
            raw_inspirations = json.loads(raw_inspirations)
        except json.JSONDecodeError as error:
            raise ValueError("inspiration is not valid serialized JSON") from error
    if not isinstance(raw_inspirations, list | tuple) or any(
        not isinstance(item, Mapping) for item in raw_inspirations
    ):
        raise ValueError("inspiration must contain a list of objects")

    inspirations = tuple(
        text
        for item in raw_inspirations
        if (text := str(item.get("insp", "")).strip())
    )
    if task == "composition" and not inspirations:
        raise ValueError("composition task requires at least one inspiration")

    student_prompt = (
        "Given the research question and background below, propose one scientifically meaningful "
        "hypothesis. Explain its central mechanism, how it differs from existing approaches, and "
        "how it could be tested.\n\n"
        f"Research question:\n{question}\n\n"
        f"Research background:\n{background}"
    )
    if task == "composition":
        formatted_inspirations = "\n".join(
            f"{index}. {inspiration}" for index, inspiration in enumerate(inspirations, start=1)
        )
        student_prompt = (
            f"{student_prompt}\n\n"
            "Use the supplied inspiration to change or extend the proposed mechanism. Explain "
            "the connection explicitly.\n\n"
            f"Supplied inspiration:\n{formatted_inspirations}"
        )

    return CanonicalExample(
        id=source_id,
        student_prompt=student_prompt,
        privileged_context=PrivilegedContext(
            historical_hypothesis=target,
            inspirations=inspirations,
        ),
        human_target=target,
        source_ids=(source_id,),
        split=split,
        task=task,
        prompt_hash=hashlib.sha256(student_prompt.encode()).hexdigest(),
    )


def select_deterministic_subset(
    records: Iterable[Mapping[str, Any]], *, size: int, seed: int
) -> tuple[Mapping[str, Any], ...]:
    """Select a stable nested subset independent of input ordering."""

    if size <= 0:
        raise ValueError("subset size must be positive")
    materialized = tuple(records)
    if size > len(materialized):
        raise ValueError(f"requested {size} records from a dataset of {len(materialized)}")

    source_ids = [str(record["source_id"]) for record in materialized]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_id values must be unique before subsetting")

    return tuple(
        sorted(
            materialized,
            key=lambda record: (
                hashlib.sha256(f"{seed}\0{record['source_id']}".encode()).digest(),
                str(record["source_id"]),
            ),
        )[:size]
    )
