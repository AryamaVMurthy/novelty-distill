"""Quality judging and deterministic semantic clustering for teacher samples."""

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JudgeSpec(BaseModel):
    """Pinned judge identity and controlled decoding settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    revision: str = Field(min_length=40, max_length=40)
    max_tokens: int = Field(default=128, gt=0)


class QualityDimensions(BaseModel):
    """Five rubric dimensions returned through SGLang structured output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relevance: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    soundness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    instruction_compliance: int = Field(ge=1, le=5)


class JudgedQuality(BaseModel):
    """Normalized quality plus audit fields from one judge request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimensions: QualityDimensions
    quality_score: float = Field(ge=0, le=1)
    request_id: str = Field(min_length=1)
    model: str = Field(min_length=1)


def build_quality_judge_payload(
    *, prompt: str, response: str, spec: JudgeSpec
) -> dict[str, Any]:
    """Build a deterministic SGLang request with a strict JSON schema."""

    if not prompt.strip() or not response.strip():
        raise ValueError("judge prompt and response must both be non-empty")
    rubric = (
        "Score the candidate scientific idea from 1 (poor) to 5 (excellent) on each "
        "requested dimension. Judge only what is supported by the supplied task and answer."
    )
    user_content = f"Task:\n{prompt}\n\nCandidate answer:\n{response}"
    return {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": rubric},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": spec.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "scientific_idea_quality",
                "strict": True,
                "schema": QualityDimensions.model_json_schema(),
            },
        },
    }


def parse_quality_judge_response(
    response: Mapping[str, Any], spec: JudgeSpec
) -> JudgedQuality:
    """Validate a structured judge response and normalize its 1--5 mean to 0--1."""

    request_id = str(response.get("id", "")).strip()
    choices = response.get("choices")
    if not request_id:
        raise ValueError("judge response has no request id")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("judge response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("judge choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("judge choice has no text content")
    try:
        dimensions = QualityDimensions.model_validate(json.loads(message["content"]))
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise ValueError("judge returned invalid structured quality scores") from error
    values = tuple(dimensions.model_dump().values())
    quality_score = (sum(values) / len(values) - 1) / 4
    return JudgedQuality(
        dimensions=dimensions,
        quality_score=quality_score,
        request_id=request_id,
        model=str(response.get("model", spec.model)),
    )


def cluster_cosine_embeddings(
    embeddings: Sequence[Sequence[float]], *, threshold: float
) -> tuple[str, ...]:
    """Return connected-component labels for a cosine-similarity threshold graph."""

    if not -1 <= threshold <= 1:
        raise ValueError("cosine threshold must be between -1 and 1")
    if not embeddings:
        raise ValueError("clustering requires at least one embedding")
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(embedding) != dimension for embedding in embeddings):
        raise ValueError("embeddings must have one shared non-zero dimension")

    normalized: list[tuple[float, ...]] = []
    for embedding in embeddings:
        norm = math.sqrt(sum(float(value) ** 2 for value in embedding))
        if norm == 0:
            raise ValueError("embeddings must be non-zero")
        normalized.append(tuple(float(value) / norm for value in embedding))

    adjacency: list[list[int]] = [[] for _ in normalized]
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            similarity = sum(
                a * b for a, b in zip(normalized[left], normalized[right], strict=True)
            )
            if similarity >= threshold:
                adjacency[left].append(right)
                adjacency[right].append(left)

    labels = [""] * len(normalized)
    cluster_index = 0
    for root in range(len(normalized)):
        if labels[root]:
            continue
        label = f"cluster-{cluster_index:03d}"
        labels[root] = label
        frontier = [root]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if not labels[neighbor]:
                    labels[neighbor] = label
                    frontier.append(neighbor)
        cluster_index += 1
    return tuple(labels)
