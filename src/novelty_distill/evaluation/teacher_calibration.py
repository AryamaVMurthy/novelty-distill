"""Configuration and prompt selection for teacher-decoding calibration."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from novelty_distill.generation.sglang import GenerationSpec


class CalibrationCondition(BaseModel):
    """One explicitly named teacher sampling condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    generation: GenerationSpec


class TeacherCalibrationStudy(BaseModel):
    """Frozen calibration matrix sharing prompts, model identity, and seeds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    prompt_count: int = Field(ge=2)
    concurrency: int = Field(gt=0)
    reference_condition: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    conditions: tuple[CalibrationCondition, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_shared_controls(self) -> "TeacherCalibrationStudy":
        ids = [condition.id for condition in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("calibration condition ids must be unique")
        if self.reference_condition not in ids:
            raise ValueError("reference condition must name one calibration condition")
        first = self.conditions[0].generation
        shared_fields = (
            "model",
            "revision",
            "samples_per_prompt",
            "seed",
            "enable_thinking",
            "top_k",
            "min_p",
        )
        for condition in self.conditions[1:]:
            for field in shared_fields:
                if getattr(condition.generation, field) != getattr(first, field):
                    raise ValueError(f"calibration conditions must share {field}")
        return self


def select_length_stratified(
    rows: Iterable[Mapping[str, Any]], *, size: int
) -> tuple[Mapping[str, Any], ...]:
    """Select deterministic midpoint representatives from equal prompt-length strata."""

    if size < 2:
        raise ValueError("calibration selection requires at least two prompts")
    materialized = tuple(rows)
    if size > len(materialized):
        raise ValueError(f"requested {size} prompts from {len(materialized)} rows")
    normalized: list[tuple[int, str, Mapping[str, Any]]] = []
    for row in materialized:
        prompt_id = str(row.get("id", "")).strip()
        prompt = row.get("student_prompt")
        if not prompt_id or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("calibration rows require non-empty id and student_prompt")
        normalized.append((len(prompt), prompt_id, row))
    ids = [item[1] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("calibration prompt ids must be unique")

    ordered = sorted(normalized, key=lambda item: (item[0], item[1]))
    selected: list[Mapping[str, Any]] = []
    total = len(ordered)
    for stratum in range(size):
        lower = stratum * total // size
        upper = (stratum + 1) * total // size
        midpoint = (lower + upper - 1) // 2
        selected.append(ordered[midpoint][2])
    return tuple(selected)
