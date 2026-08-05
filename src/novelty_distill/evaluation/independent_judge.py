"""Blinded, stratified calibration of an independent model judge."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from statistics import fmean, pstdev
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from novelty_distill.evaluation.teacher_annotation import QualityDimensions

QUALITY_DIMENSIONS = (
    "relevance",
    "feasibility",
    "soundness",
    "clarity",
    "instruction_compliance",
)

_RUBRIC = (
    "You are independently auditing a candidate scientific hypothesis and test plan. "
    "Use only the supplied task and candidate answer; do not infer the generator identity. "
    "Score each dimension independently from 1 to 5. Relevance: directly addresses the "
    "question and background. Feasibility: the intervention, controls, measurements, and "
    "resources form an executable test. Soundness: the mechanism is coherent and the claimed "
    "conclusion does not outrun the proposed evidence. Clarity: specific and unambiguous. "
    "Instruction compliance: all explicit response requirements are met. Reserve 5 for an "
    "exceptional answer with no material gap; 4 is strong with a minor gap; 3 is plausible but "
    "has a substantive gap; 2 has major defects; 1 is missing, contradictory, or off-task. "
    "Do not reward length or fluency by itself. Set fatal_flaw when a central biological, "
    "physical, ethical, or experimental impossibility invalidates the plan. The rationale must "
    "name the most important concrete strength or defect. This is not a global novelty search, "
    "so do not claim that the idea is novel in the scientific literature."
)

_OUTPUT_INSTRUCTION = (
    "Return only a compact JSON object with exactly these keys: relevance, feasibility, "
    "soundness, clarity, and instruction_compliance as integers; fatal_flaw as a boolean; "
    "and brief_rationale as a string of at most 40 words. Do not output Markdown or any text "
    "outside the JSON object."
)


class CalibrationEntry(BaseModel):
    """One private calibration item; only prompt and answer are sent to the judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blind_id: str = Field(min_length=16, max_length=32)
    method: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    prompt: str = Field(min_length=1)
    text: str = Field(min_length=1)
    text_sha256: str = Field(min_length=64, max_length=64)
    qwen_dimensions: dict[str, int]
    qwen_core_score: float = Field(ge=1, le=5)
    score_stratum: int = Field(ge=0, le=3)
    repeat_of: str | None = None


class ExternalJudgeScore(BaseModel):
    """Strict response schema for the independent judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relevance: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    soundness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    instruction_compliance: int = Field(ge=1, le=5)
    fatal_flaw: bool
    brief_rationale: str = Field(min_length=1, max_length=500)


class ExternalJudgeResponse(BaseModel):
    """Validated response plus provider audit fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    score: ExternalJudgeScore
    usage: dict[str, int]


def _hash_order(seed: int, *parts: object) -> str:
    value = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode()).hexdigest()


def _stratum_quotas(total: int) -> tuple[int, int, int, int]:
    base, remainder = divmod(total, 4)
    return tuple(base + int(index < remainder) for index in range(4))  # type: ignore[return-value]


def _balanced_repeat_sources(
    originals: Sequence[CalibrationEntry], *, repeat_fraction: float, seed: int
) -> tuple[CalibrationEntry, ...]:
    """Choose hidden repeats with deterministic near-equal method representation."""

    repeat_count = round(len(originals) * repeat_fraction)
    if repeat_count == 0:
        return ()
    by_method: dict[str, list[CalibrationEntry]] = {}
    for entry in originals:
        by_method.setdefault(entry.method, []).append(entry)
    methods = sorted(by_method, key=lambda method: _hash_order(seed, "method", method))
    base, remainder = divmod(repeat_count, len(methods))
    selected: list[CalibrationEntry] = []
    for index, method in enumerate(methods):
        quota = base + int(index < remainder)
        candidates = sorted(
            by_method[method],
            key=lambda entry: _hash_order(seed, "repeat-source", entry.blind_id),
        )
        if quota > len(candidates):
            raise ValueError(f"method {method} has too few originals for repeat quota")
        selected.extend(candidates[:quota])
    return tuple(sorted(selected, key=lambda entry: _hash_order(seed, entry.blind_id)))


def _build_unpaired_calibration_sample(
    *,
    candidates_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    prompts: Mapping[str, str],
    samples_per_method: int,
    repeat_fraction: float,
    seed: int,
) -> tuple[CalibrationEntry, ...]:
    """Rank-stratify each method by the old judge score and add blinded repeats."""

    if not candidates_by_method:
        raise ValueError("at least one method is required")
    if samples_per_method < 4:
        raise ValueError("samples_per_method must be at least four")
    if not 0 <= repeat_fraction <= 1:
        raise ValueError("repeat_fraction must be between zero and one")

    selected: list[CalibrationEntry] = []
    for method, raw_candidates in sorted(candidates_by_method.items()):
        if not method.strip():
            raise ValueError("method names must be non-empty")
        materialized: list[dict[str, Any]] = []
        identities: set[tuple[str, int]] = set()
        for raw in raw_candidates:
            prompt_id = str(raw.get("prompt_id", "")).strip()
            sample_index = int(raw.get("sample_index", -1))
            identity = (prompt_id, sample_index)
            if identity in identities:
                raise ValueError(f"duplicate candidate {method}/{prompt_id}/{sample_index}")
            identities.add(identity)
            if prompt_id not in prompts:
                raise ValueError(f"candidate prompt {prompt_id} is absent from prompts")
            text = str(raw.get("text", "")).strip()
            if not text:
                raise ValueError(f"candidate {method}/{prompt_id}/{sample_index} has empty text")
            dimensions = QualityDimensions.model_validate(raw.get("dimensions"))
            dimension_dict = dimensions.model_dump(mode="json")
            core_score = fmean((dimension_dict["feasibility"], dimension_dict["soundness"]))
            materialized.append(
                {
                    "prompt_id": prompt_id,
                    "sample_index": sample_index,
                    "text": text,
                    "dimensions": dimension_dict,
                    "core_score": core_score,
                }
            )
        if len(materialized) < samples_per_method:
            raise ValueError(
                f"method {method} has {len(materialized)} candidates, needs {samples_per_method}"
            )

        ranked = sorted(
            materialized,
            key=lambda item: (
                item["core_score"],
                _hash_order(seed, method, item["prompt_id"], item["sample_index"]),
            ),
        )
        strata: list[list[dict[str, Any]]] = [[], [], [], []]
        for rank, item in enumerate(ranked):
            stratum = min(3, rank * 4 // len(ranked))
            strata[stratum].append(item)
        for stratum, quota in enumerate(_stratum_quotas(samples_per_method)):
            if len(strata[stratum]) < quota:
                raise ValueError(f"method {method} score stratum {stratum} is too small")
            choices = sorted(
                strata[stratum],
                key=lambda item: _hash_order(
                    seed + 1, method, item["prompt_id"], item["sample_index"]
                ),
            )[:quota]
            for item in choices:
                text_hash = hashlib.sha256(item["text"].encode()).hexdigest()
                blind_id = (
                    "cal-"
                    + _hash_order(
                        seed,
                        "original",
                        method,
                        item["prompt_id"],
                        item["sample_index"],
                        text_hash,
                    )[:20]
                )
                selected.append(
                    CalibrationEntry(
                        blind_id=blind_id,
                        method=method,
                        prompt_id=item["prompt_id"],
                        sample_index=item["sample_index"],
                        prompt=prompts[item["prompt_id"]],
                        text=item["text"],
                        text_sha256=text_hash,
                        qwen_dimensions=item["dimensions"],
                        qwen_core_score=item["core_score"],
                        score_stratum=stratum,
                    )
                )

    originals = sorted(selected, key=lambda entry: _hash_order(seed + 2, entry.blind_id))
    repeat_sources = _balanced_repeat_sources(
        originals, repeat_fraction=repeat_fraction, seed=seed + 3
    )
    repeats = [
        entry.model_copy(
            update={
                "blind_id": "cal-" + _hash_order(seed, "repeat", entry.blind_id, index)[:20],
                "repeat_of": entry.blind_id,
            }
        )
        for index, entry in enumerate(repeat_sources)
    ]
    return tuple(
        sorted((*originals, *repeats), key=lambda entry: _hash_order(seed + 4, entry.blind_id))
    )


def build_blinded_calibration_sample(
    *,
    candidates_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    prompts: Mapping[str, str],
    samples_per_method: int,
    repeat_fraction: float,
    seed: int,
) -> tuple[CalibrationEntry, ...]:
    """Select paired score-stratified slots across methods and add blinded repeats."""

    if not candidates_by_method:
        raise ValueError("at least one method is required")
    if samples_per_method < 4:
        raise ValueError("samples_per_method must be at least four")
    if not 0 <= repeat_fraction <= 1:
        raise ValueError("repeat_fraction must be between zero and one")

    by_method: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for method, raw_candidates in sorted(candidates_by_method.items()):
        if not method.strip():
            raise ValueError("method names must be non-empty")
        materialized: dict[tuple[str, int], dict[str, Any]] = {}
        for raw in raw_candidates:
            prompt_id = str(raw.get("prompt_id", "")).strip()
            sample_index = int(raw.get("sample_index", -1))
            identity = (prompt_id, sample_index)
            if identity in materialized:
                raise ValueError(f"duplicate candidate {method}/{prompt_id}/{sample_index}")
            if prompt_id not in prompts:
                raise ValueError(f"candidate prompt {prompt_id} is absent from prompts")
            text = str(raw.get("text", "")).strip()
            if not text:
                raise ValueError(f"candidate {method}/{prompt_id}/{sample_index} has empty text")
            dimensions = QualityDimensions.model_validate(raw.get("dimensions")).model_dump(
                mode="json"
            )
            materialized[identity] = {
                "prompt_id": prompt_id,
                "sample_index": sample_index,
                "text": text,
                "dimensions": dimensions,
                "core_score": fmean((dimensions["feasibility"], dimensions["soundness"])),
            }
        if len(materialized) < samples_per_method:
            raise ValueError(
                f"method {method} has {len(materialized)} candidates, needs {samples_per_method}"
            )
        by_method[method] = materialized

    common = set.intersection(*(set(candidates) for candidates in by_method.values()))
    if len(common) < samples_per_method:
        raise ValueError(
            f"only {len(common)} paired candidate slots exist, needs {samples_per_method}"
        )
    common_prompt_count = len({prompt_id for prompt_id, _ in common})
    if common_prompt_count < samples_per_method:
        raise ValueError(
            f"only {common_prompt_count} unique common prompts exist, needs {samples_per_method}"
        )
    methods = sorted(by_method)
    ranked = sorted(
        (
            (
                identity,
                fmean(by_method[method][identity]["core_score"] for method in methods),
            )
            for identity in common
        ),
        key=lambda item: (
            item[1],
            _hash_order(seed, "pooled", item[0][0], item[0][1]),
        ),
    )
    strata: list[list[tuple[str, int]]] = [[], [], [], []]
    for rank, (identity, _) in enumerate(ranked):
        strata[min(3, rank * 4 // len(ranked))].append(identity)

    # Select exactly one sample slot per prompt while retaining fixed stratum
    # quotas. A prompt can have samples in more than one score stratum, so this
    # is a small deterministic bipartite matching problem rather than four
    # independent list slices.
    candidates_by_stratum: dict[int, list[tuple[str, int]]] = {}
    for stratum, identities in enumerate(strata):
        preferred_by_prompt: dict[str, tuple[str, int]] = {}
        for identity in identities:
            prompt_id = identity[0]
            incumbent = preferred_by_prompt.get(prompt_id)
            if incumbent is None or _hash_order(
                seed + 1, "paired", identity[0], identity[1]
            ) < _hash_order(seed + 1, "paired", incumbent[0], incumbent[1]):
                preferred_by_prompt[prompt_id] = identity
        candidates_by_stratum[stratum] = sorted(
            preferred_by_prompt.values(),
            key=lambda identity: _hash_order(seed + 1, "paired", identity[0], identity[1]),
        )

    demands = [
        (stratum, offset)
        for stratum, quota in enumerate(_stratum_quotas(samples_per_method))
        for offset in range(quota)
    ]
    prompt_to_demand: dict[str, tuple[int, int]] = {}
    demand_to_identity: dict[tuple[int, int], tuple[str, int]] = {}

    def assign(demand: tuple[int, int], seen_prompts: set[str]) -> bool:
        stratum, _ = demand
        for identity in candidates_by_stratum[stratum]:
            prompt_id = identity[0]
            if prompt_id in seen_prompts:
                continue
            seen_prompts.add(prompt_id)
            previous = prompt_to_demand.get(prompt_id)
            if previous is None or assign(previous, seen_prompts):
                prompt_to_demand[prompt_id] = demand
                demand_to_identity[demand] = identity
                return True
        return False

    for demand in sorted(
        demands,
        key=lambda item: (len(candidates_by_stratum[item[0]]), item[0], item[1]),
    ):
        if not assign(demand, set()):
            raise ValueError(
                "could not satisfy pooled score-stratum quotas with unique common prompts"
            )
    selected_slots = [(demand_to_identity[demand], demand[0]) for demand in sorted(demands)]

    selected: list[CalibrationEntry] = []
    for identity, stratum in selected_slots:
        for method in methods:
            item = by_method[method][identity]
            text_hash = hashlib.sha256(item["text"].encode()).hexdigest()
            blind_id = (
                "cal-"
                + _hash_order(
                    seed,
                    "original",
                    method,
                    item["prompt_id"],
                    item["sample_index"],
                    text_hash,
                )[:20]
            )
            selected.append(
                CalibrationEntry(
                    blind_id=blind_id,
                    method=method,
                    prompt_id=item["prompt_id"],
                    sample_index=item["sample_index"],
                    prompt=prompts[item["prompt_id"]],
                    text=item["text"],
                    text_sha256=text_hash,
                    qwen_dimensions=item["dimensions"],
                    qwen_core_score=item["core_score"],
                    score_stratum=stratum,
                )
            )

    originals = sorted(selected, key=lambda entry: _hash_order(seed + 2, entry.blind_id))
    repeat_sources = _balanced_repeat_sources(
        originals, repeat_fraction=repeat_fraction, seed=seed + 3
    )
    repeats = [
        entry.model_copy(
            update={
                "blind_id": "cal-" + _hash_order(seed, "repeat", entry.blind_id, index)[:20],
                "repeat_of": entry.blind_id,
            }
        )
        for index, entry in enumerate(repeat_sources)
    ]
    return tuple(
        sorted((*originals, *repeats), key=lambda entry: _hash_order(seed + 4, entry.blind_id))
    )


def independent_judge_protocol_hash() -> str:
    """Content identity for the exact rubric and structured response schema."""

    payload = {
        "rubric": _RUBRIC,
        "output_instruction": _OUTPUT_INSTRUCTION,
        "response_parser": "raw-json-or-single-json-code-fence-v1",
        "response_format": {"type": "json_object"},
        "schema": ExternalJudgeScore.model_json_schema(),
        "version": 3,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_independent_judge_payload(
    entry: CalibrationEntry, *, model: str, max_tokens: int = 256
) -> dict[str, Any]:
    """Build a zero-temperature request that does not expose private method metadata."""

    if not model.strip() or max_tokens <= 0:
        raise ValueError("model and positive max_tokens are required")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": f"{_RUBRIC} {_OUTPUT_INSTRUCTION}"},
            {
                "role": "user",
                "content": f"Task:\n{entry.prompt}\n\nCandidate answer:\n{entry.text}",
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def parse_independent_judge_response(response: Mapping[str, Any]) -> ExternalJudgeResponse:
    """Validate one OpenAI-compatible structured judge response."""

    request_id = str(response.get("id", "")).strip()
    choices = response.get("choices")
    if not request_id:
        raise ValueError("independent judge response has no request id")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("independent judge response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("independent judge choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("independent judge choice has no text content")
    content = message["content"].strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n```", content, flags=re.DOTALL)
    if fenced is not None:
        content = fenced.group(1)
    try:
        score = ExternalJudgeScore.model_validate(json.loads(content))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("independent judge returned invalid structured scores") from error
    usage = response.get("usage", {})
    if not isinstance(usage, Mapping):
        raise ValueError("independent judge usage must be an object")
    parsed_usage = {
        str(key): int(value)
        for key, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return ExternalJudgeResponse(
        request_id=request_id,
        model=str(response.get("model", "unknown")),
        score=score,
        usage=parsed_usage,
    )


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=False))
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ranked):
        end = cursor + 1
        while end < len(ranked) and ranked[end][1] == ranked[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in range(cursor, end):
            result[ranked[index][0]] = rank
        cursor = end
    return result


def _score_distribution(values: Sequence[float]) -> dict[str, Any]:
    """Summarize ordinal-score saturation without treating the scale as ground truth."""

    if not values:
        raise ValueError("score-distribution values must be non-empty")
    return {
        "histogram": {str(score): sum(value == score for value in values) for score in range(1, 6)},
        "mean": fmean(values),
        "population_sd": pstdev(values),
        "unique_scores": len(set(values)),
        "floor_rate": fmean(float(value == 1) for value in values),
        "ceiling_rate": fmean(float(value == 5) for value in values),
    }


def analyze_independent_judgments(
    *, entries: Sequence[CalibrationEntry], ratings: Mapping[str, ExternalJudgeScore]
) -> dict[str, Any]:
    """Compare independent scores to Qwen and quantify hidden-repeat reliability."""

    if not entries or len({entry.blind_id for entry in entries}) != len(entries):
        raise ValueError("calibration entries must be non-empty with unique blind ids")
    if set(ratings) != {entry.blind_id for entry in entries}:
        raise ValueError("ratings must exactly cover all calibration blind ids")
    originals = [entry for entry in entries if entry.repeat_of is None]
    repeats = [entry for entry in entries if entry.repeat_of is not None]
    original_by_id = {entry.blind_id: entry for entry in originals}
    if any(entry.repeat_of not in original_by_id for entry in repeats):
        raise ValueError("repeat points to an absent original")

    agreement: dict[str, Any] = {}
    score_distributions: dict[str, Any] = {}
    for dimension in QUALITY_DIMENSIONS:
        qwen = [float(entry.qwen_dimensions[dimension]) for entry in originals]
        external = [float(getattr(ratings[entry.blind_id], dimension)) for entry in originals]
        differences = [right - left for left, right in zip(qwen, external, strict=False)]
        agreement[dimension] = {
            "qwen_mean": fmean(qwen),
            "independent_mean": fmean(external),
            "independent_minus_qwen_mean": fmean(differences),
            "exact_rate": fmean(
                float(left == right) for left, right in zip(qwen, external, strict=False)
            ),
            "within_one_rate": fmean(
                float(abs(left - right) <= 1) for left, right in zip(qwen, external, strict=False)
            ),
            "mean_absolute_difference": fmean(abs(value) for value in differences),
            "pearson": _pearson(qwen, external),
            "spearman": _pearson(_average_ranks(qwen), _average_ranks(external)),
        }
        score_distributions[dimension] = {
            "qwen": _score_distribution(qwen),
            "independent": _score_distribution(external),
        }

    rationales = [ratings[entry.blind_id].brief_rationale.strip() for entry in originals]
    rationale_word_counts = [len(rationale.split()) for rationale in rationales]
    rationale_diagnostics = {
        "unique_exact_rationales": len(set(rationales)),
        "exact_duplicate_rate": 1 - len(set(rationales)) / len(rationales),
        "mean_word_count": fmean(rationale_word_counts),
        "max_word_count": max(rationale_word_counts),
        "over_40_word_rate": fmean(float(count > 40) for count in rationale_word_counts),
    }

    repeat_reliability: dict[str, Any] = {}
    for dimension in QUALITY_DIMENSIONS:
        original_values = [float(getattr(ratings[entry.repeat_of], dimension)) for entry in repeats]
        repeated_values = [float(getattr(ratings[entry.blind_id], dimension)) for entry in repeats]
        repeat_reliability[dimension] = {
            "exact_rate": fmean(
                float(left == right)
                for left, right in zip(original_values, repeated_values, strict=False)
            )
            if repeats
            else None,
            "within_one_rate": fmean(
                float(abs(left - right) <= 1)
                for left, right in zip(original_values, repeated_values, strict=False)
            )
            if repeats
            else None,
            "mean_absolute_difference": fmean(
                abs(left - right)
                for left, right in zip(original_values, repeated_values, strict=False)
            )
            if repeats
            else None,
        }

    method_means: dict[str, Any] = {}
    for method in sorted({entry.method for entry in originals}):
        method_entries = [entry for entry in originals if entry.method == method]
        method_means[method] = {
            dimension: fmean(
                float(getattr(ratings[entry.blind_id], dimension)) for entry in method_entries
            )
            for dimension in QUALITY_DIMENSIONS
        }
        method_means[method]["fatal_flaw_rate"] = fmean(
            float(ratings[entry.blind_id].fatal_flaw) for entry in method_entries
        )

    by_slot = {(entry.method, entry.prompt_id, entry.sample_index): entry for entry in originals}
    slots = sorted({(entry.prompt_id, entry.sample_index) for entry in originals})
    methods = sorted({entry.method for entry in originals})
    if len(by_slot) != len(slots) * len(methods):
        raise ValueError("original calibration entries are not a complete paired design")
    paired_contrasts: dict[str, Any] = {}
    for left_index, left_method in enumerate(methods):
        for right_method in methods[left_index + 1 :]:
            contrast: dict[str, Any] = {}
            for dimension in ("feasibility", "soundness"):
                differences = [
                    float(
                        getattr(
                            ratings[by_slot[(right_method, prompt_id, sample_index)].blind_id],
                            dimension,
                        )
                    )
                    - float(
                        getattr(
                            ratings[by_slot[(left_method, prompt_id, sample_index)].blind_id],
                            dimension,
                        )
                    )
                    for prompt_id, sample_index in slots
                ]
                bootstrap_rng = random.Random(
                    int(
                        _hash_order(20260805, left_method, right_method, dimension)[:16],
                        16,
                    )
                )
                bootstrap_means = sorted(
                    fmean(bootstrap_rng.choice(differences) for _ in differences)
                    for _ in range(5000)
                )
                contrast[dimension] = {
                    "mean_difference": fmean(differences),
                    "ci95_percentile": [
                        bootstrap_means[124],
                        bootstrap_means[4874],
                    ],
                    "right_win_rate": fmean(float(value > 0) for value in differences),
                    "tie_rate": fmean(float(value == 0) for value in differences),
                    "paired_slots": len(differences),
                }
            paired_contrasts[f"{right_method}_minus_{left_method}"] = contrast

    return {
        "schema_version": 1,
        "counts": {
            "originals": len(originals),
            "repeats": len(repeats),
            "total": len(entries),
        },
        "agreement": agreement,
        "score_distributions": score_distributions,
        "rationale_diagnostics": rationale_diagnostics,
        "repeat_reliability": repeat_reliability,
        "method_means": method_means,
        "paired_contrasts": paired_contrasts,
        "claim_boundary": (
            "This different-family judge is a sensitivity analysis, not ground truth and not "
            "a measurement of global scientific novelty. Human calibration remains required."
        ),
    }
