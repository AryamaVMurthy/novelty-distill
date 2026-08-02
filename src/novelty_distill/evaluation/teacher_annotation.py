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
        "Score dimensions independently from 1 to 5 and judge only evidence in the supplied "
        "task and candidate answer. Relevance means directly answering the research question "
        "and using its background. Feasibility means an actionable test with credible "
        "measurements and resources. Soundness means a coherent mechanism and conclusions "
        "that do not outrun the proposed evidence. Clarity means specific, unambiguous prose. "
        "Instruction compliance includes every response requirement in the task. Reserve 5 "
        "for an exceptional answer with a concrete mechanism and operational test and no "
        "material gap; 4 is strong with a minor gap; 3 means plausible but has a substantive "
        "gap or vagueness; 2 has major defects; 1 means the dimension is missing, contradictory, "
        "or off-task. Do not default to 5 merely because the answer is fluent or long."
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
    """Return deterministic complete-linkage labels at a cosine-similarity threshold."""

    if not -1 <= threshold <= 1:
        raise ValueError("cosine threshold must be between -1 and 1")
    normalized = _normalize_embeddings(embeddings)
    similarities = [[1.0] * len(normalized) for _ in normalized]
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            similarity = sum(
                a * b for a, b in zip(normalized[left], normalized[right], strict=True)
            )
            similarities[left][right] = similarity
            similarities[right][left] = similarity

    clusters = [(index,) for index in range(len(normalized))]
    while True:
        candidates: list[tuple[float, tuple[int, ...], tuple[int, ...], int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                complete_similarity = min(
                    similarities[left_index][right_index]
                    for left_index in clusters[left]
                    for right_index in clusters[right]
                )
                if complete_similarity >= threshold:
                    candidates.append(
                        (
                            -complete_similarity,
                            clusters[left],
                            clusters[right],
                            left,
                            right,
                        )
                    )
        if not candidates:
            break
        _, _, _, left, right = min(candidates)
        merged = tuple(sorted((*clusters[left], *clusters[right])))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left, right}
        ]
        clusters.append(merged)
        clusters.sort()

    labels = [""] * len(normalized)
    for cluster_index, cluster in enumerate(clusters):
        label = f"cluster-{cluster_index:03d}"
        for index in cluster:
            labels[index] = label
    return tuple(labels)


def cosine_embedding_diagnostics(
    embeddings: Sequence[Sequence[float]], *, thresholds: Sequence[float]
) -> dict[str, Any]:
    """Summarize pairwise cosine values and cluster-count sensitivity."""

    normalized = _normalize_embeddings(embeddings)
    if len(normalized) < 2:
        raise ValueError("cosine diagnostics require at least two embeddings")
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("diagnostic thresholds must be non-empty and unique")
    similarities = [
        sum(a * b for a, b in zip(normalized[left], normalized[right], strict=True))
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    ]
    clusters_by_threshold = {
        f"{threshold:.3f}": len(
            set(cluster_cosine_embeddings(normalized, threshold=threshold))
        )
        for threshold in thresholds
    }
    return {
        "pair_count": len(similarities),
        "cosine_min": min(similarities),
        "cosine_mean": sum(similarities) / len(similarities),
        "cosine_max": max(similarities),
        "clusters_by_threshold": clusters_by_threshold,
    }


def _normalize_embeddings(
    embeddings: Sequence[Sequence[float]],
) -> list[tuple[float, ...]]:
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
    return normalized
