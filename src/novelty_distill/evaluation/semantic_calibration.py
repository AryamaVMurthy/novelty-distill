"""Blinded human calibration for embedding-based semantic equivalence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PrivateSemanticPair(BaseModel):
    """Source key withheld from raters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blind_id: str = Field(min_length=16, max_length=32)
    prompt_id: str = Field(min_length=1)
    left_index: int = Field(ge=0)
    right_index: int = Field(ge=0)
    source_pair_sha256: str = Field(min_length=64, max_length=64)
    similarity: float = Field(ge=-1, le=1.001)
    similarity_bin: int = Field(ge=0)
    repeat_of: str | None = None


class PublicSemanticPair(BaseModel):
    """Only this view is shown to human raters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blind_id: str = Field(min_length=16, max_length=32)
    task: str = Field(min_length=1)
    answer_a: str = Field(min_length=1)
    answer_b: str = Field(min_length=1)
    instruction: str = Field(min_length=1)


class SemanticCalibrationSample(BaseModel):
    """Parallel private/public views with deterministic order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    private_entries: tuple[PrivateSemanticPair, ...]
    public_entries: tuple[PublicSemanticPair, ...]


class HumanEquivalenceLabel(BaseModel):
    """One source-blinded human judgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blind_id: str = Field(min_length=16, max_length=32)
    label: Literal["equivalent", "not_equivalent", "uncertain"]
    confidence: int = Field(ge=1, le=3)
    rationale: str = Field(default="", max_length=1000)


_RATER_INSTRUCTION = (
    "Decide whether Answer A and Answer B express the same central scientific idea for the "
    "supplied task. Mark equivalent only when their central mechanism, intervention or object "
    "of study, and decisive experimental test would lead to the same substantive research "
    "project. Differences in wording, detail, controls, or measurement may still be equivalent. "
    "Mark not_equivalent when any central mechanism, intervention, causal claim, or decisive test "
    "differs. Use uncertain only when the text is too incomplete or ambiguous to decide. Do not "
    "judge quality, feasibility, or global novelty."
)


def semantic_calibration_protocol_hash() -> str:
    """Content identity for the rater contract and label schema."""

    payload = {
        "version": 1,
        "rater_instruction": _RATER_INSTRUCTION,
        "label_schema": HumanEquivalenceLabel.model_json_schema(),
        "selection_rule": "maximize_balanced_accuracy_ties_choose_higher_threshold",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash(seed: int, *parts: object) -> str:
    raw = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(raw.encode()).hexdigest()


def _validate_bins(bins: Sequence[tuple[float, float]]) -> None:
    if not bins:
        raise ValueError("similarity bins must be non-empty")
    for index, (lower, upper) in enumerate(bins):
        if not -1 <= lower < upper <= 1.001:
            raise ValueError("similarity bins must have valid increasing bounds")
        if index and lower < bins[index - 1][1]:
            raise ValueError("similarity bins must not overlap")


def _pair_bin(similarity: float, bins: Sequence[tuple[float, float]]) -> int | None:
    for index, (lower, upper) in enumerate(bins):
        if lower <= similarity < upper:
            return index
    return None


def build_semantic_calibration_sample(
    *,
    pairs: Sequence[Mapping[str, Any]],
    prompts: Mapping[str, str],
    similarity_bins: Sequence[tuple[float, float]],
    pairs_per_bin: int,
    repeat_fraction: float,
    seed: int,
) -> SemanticCalibrationSample:
    """Select one pair per prompt across similarity strata, then add reversed repeats."""

    _validate_bins(similarity_bins)
    if pairs_per_bin <= 0:
        raise ValueError("pairs_per_bin must be positive")
    if not 0 <= repeat_fraction <= 1:
        raise ValueError("repeat_fraction must be between zero and one")
    by_bin: list[list[dict[str, Any]]] = [[] for _ in similarity_bins]
    identities: set[tuple[str, int, int]] = set()
    for raw in pairs:
        prompt_id = str(raw.get("prompt_id", "")).strip()
        left_index = int(raw.get("left_index", -1))
        right_index = int(raw.get("right_index", -1))
        if left_index == right_index or min(left_index, right_index) < 0:
            raise ValueError("semantic pair indices must be distinct and non-negative")
        identity = (prompt_id, min(left_index, right_index), max(left_index, right_index))
        if identity in identities:
            raise ValueError(f"duplicate semantic pair {identity}")
        identities.add(identity)
        if prompt_id not in prompts:
            raise ValueError(f"semantic pair prompt {prompt_id} is absent from prompts")
        left_text = str(raw.get("left_text", "")).strip()
        right_text = str(raw.get("right_text", "")).strip()
        if not left_text or not right_text or left_text == right_text:
            raise ValueError("semantic pair answers must be non-empty and distinct")
        similarity = float(raw.get("similarity", math.nan))
        if not math.isfinite(similarity):
            raise ValueError("semantic pair similarity must be finite")
        bin_index = _pair_bin(similarity, similarity_bins)
        if bin_index is not None:
            by_bin[bin_index].append(
                {
                    "prompt_id": prompt_id,
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_text": left_text,
                    "right_text": right_text,
                    "similarity": similarity,
                    "bin_index": bin_index,
                }
            )

    used_prompts: set[str] = set()
    chosen_by_bin: dict[int, list[dict[str, Any]]] = {}
    for bin_index in sorted(range(len(by_bin)), key=lambda index: len(by_bin[index])):
        ordered = sorted(
            by_bin[bin_index],
            key=lambda item: _hash(
                seed,
                "pair",
                item["prompt_id"],
                item["left_index"],
                item["right_index"],
            ),
        )
        chosen: list[dict[str, Any]] = []
        chosen_prompts: set[str] = set()
        for item in ordered:
            prompt_id = item["prompt_id"]
            if prompt_id not in used_prompts and prompt_id not in chosen_prompts:
                chosen.append(item)
                chosen_prompts.add(prompt_id)
                if len(chosen) == pairs_per_bin:
                    break
        if len(chosen) != pairs_per_bin:
            raise ValueError(
                f"similarity bin {bin_index} has only {len(chosen)} globally unique prompts; "
                f"needs {pairs_per_bin}"
            )
        chosen_by_bin[bin_index] = chosen
        used_prompts.update(item["prompt_id"] for item in chosen)

    private_entries: list[PrivateSemanticPair] = []
    public_entries: list[PublicSemanticPair] = []
    public_by_id: dict[str, PublicSemanticPair] = {}
    for bin_index in range(len(similarity_bins)):
        for item in chosen_by_bin[bin_index]:
            pair_hash = hashlib.sha256(
                (item["prompt_id"] + "\0" + item["left_text"] + "\0" + item["right_text"]).encode()
            ).hexdigest()
            blind_id = "sem-" + _hash(seed, "original", pair_hash)[:20]
            swap = int(_hash(seed, "order", pair_hash), 16) % 2 == 1
            answer_a, answer_b = (
                (item["right_text"], item["left_text"])
                if swap
                else (item["left_text"], item["right_text"])
            )
            private = PrivateSemanticPair(
                blind_id=blind_id,
                prompt_id=item["prompt_id"],
                left_index=item["left_index"],
                right_index=item["right_index"],
                source_pair_sha256=pair_hash,
                similarity=item["similarity"],
                similarity_bin=bin_index,
            )
            public = PublicSemanticPair(
                blind_id=blind_id,
                task=prompts[item["prompt_id"]],
                answer_a=answer_a,
                answer_b=answer_b,
                instruction=_RATER_INSTRUCTION,
            )
            private_entries.append(private)
            public_entries.append(public)
            public_by_id[blind_id] = public

    originals = sorted(private_entries, key=lambda entry: _hash(seed + 1, entry.blind_id))
    repeat_count = round(len(originals) * repeat_fraction)
    repeat_sources = sorted(originals, key=lambda entry: _hash(seed + 2, entry.blind_id))[
        :repeat_count
    ]
    for index, source in enumerate(repeat_sources):
        blind_id = "sem-" + _hash(seed, "repeat", source.blind_id, index)[:20]
        source_public = public_by_id[source.blind_id]
        private_entries.append(
            source.model_copy(update={"blind_id": blind_id, "repeat_of": source.blind_id})
        )
        public_entries.append(
            source_public.model_copy(
                update={
                    "blind_id": blind_id,
                    "answer_a": source_public.answer_b,
                    "answer_b": source_public.answer_a,
                }
            )
        )

    public_lookup = {entry.blind_id: entry for entry in public_entries}
    private_lookup = {entry.blind_id: entry for entry in private_entries}
    order = sorted(private_lookup, key=lambda blind_id: _hash(seed + 3, blind_id))
    return SemanticCalibrationSample(
        private_entries=tuple(private_lookup[blind_id] for blind_id in order),
        public_entries=tuple(public_lookup[blind_id] for blind_id in order),
    )


def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def analyze_semantic_calibration(
    *,
    entries: Sequence[PrivateSemanticPair],
    rater_one: Mapping[str, HumanEquivalenceLabel],
    rater_two: Mapping[str, HumanEquivalenceLabel],
    adjudicated: Mapping[str, Literal["equivalent", "not_equivalent"]],
    threshold_grid: Sequence[float],
) -> dict[str, Any]:
    """Resolve two-rater labels and select a threshold by balanced accuracy."""

    if not entries or len({entry.blind_id for entry in entries}) != len(entries):
        raise ValueError("semantic calibration entries must be unique and non-empty")
    expected_ids = {entry.blind_id for entry in entries}
    for name, labels in (("rater_one", rater_one), ("rater_two", rater_two)):
        if set(labels) != expected_ids or any(
            key != value.blind_id for key, value in labels.items()
        ):
            raise ValueError(f"{name} labels must exactly cover and identify every blind id")
    if (
        not threshold_grid
        or any(not -1 <= value <= 1 for value in threshold_grid)
        or tuple(sorted(set(threshold_grid))) != tuple(threshold_grid)
    ):
        raise ValueError("threshold grid must be unique, sorted, non-empty, and bounded")

    originals = [entry for entry in entries if entry.repeat_of is None]
    repeats = [entry for entry in entries if entry.repeat_of is not None]
    original_ids = {entry.blind_id for entry in originals}
    if any(entry.repeat_of not in original_ids for entry in repeats):
        raise ValueError("semantic calibration repeat points to an absent original")
    left_labels = [rater_one[entry.blind_id].label for entry in originals]
    right_labels = [rater_two[entry.blind_id].label for entry in originals]
    resolved: dict[str, str] = {}
    disagreement_ids: list[str] = []
    for entry in originals:
        left = rater_one[entry.blind_id].label
        right = rater_two[entry.blind_id].label
        if left == right and left != "uncertain":
            resolved[entry.blind_id] = left
        else:
            disagreement_ids.append(entry.blind_id)
            if entry.blind_id not in adjudicated:
                raise ValueError(
                    f"adjudication is required for disagreement or uncertainty {entry.blind_id}"
                )
            resolved[entry.blind_id] = adjudicated[entry.blind_id]
    if set(adjudicated) != set(disagreement_ids):
        raise ValueError("adjudication keys must exactly match disagreements and uncertainties")

    repeat_reliability: dict[str, Any] = {}
    for name, labels in (("rater_one", rater_one), ("rater_two", rater_two)):
        exact = [labels[entry.blind_id].label == labels[entry.repeat_of].label for entry in repeats]
        repeat_reliability[f"{name}_exact_rate"] = sum(exact) / len(exact) if exact else None

    truth = [resolved[entry.blind_id] == "equivalent" for entry in originals]
    if not any(truth) or all(truth):
        raise ValueError("adjudicated labels must contain both equivalence classes")
    threshold_metrics: dict[str, Any] = {}
    for threshold in threshold_grid:
        predicted = [entry.similarity >= threshold for entry in originals]
        true_positive = sum(
            predict and actual for predict, actual in zip(predicted, truth, strict=True)
        )
        true_negative = sum(
            not predict and not actual for predict, actual in zip(predicted, truth, strict=True)
        )
        false_positive = sum(
            predict and not actual for predict, actual in zip(predicted, truth, strict=True)
        )
        false_negative = sum(
            not predict and actual for predict, actual in zip(predicted, truth, strict=True)
        )
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        threshold_metrics[f"{threshold:.3f}"] = {
            "threshold": threshold,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "precision": precision,
        }
    selected = max(
        threshold_grid,
        key=lambda threshold: (
            threshold_metrics[f"{threshold:.3f}"]["balanced_accuracy"],
            threshold,
        ),
    )
    kappa = _cohen_kappa(left_labels, right_labels)
    return {
        "schema_version": 1,
        "counts": {
            "originals": len(originals),
            "repeats": len(repeats),
            "adjudicated": len(disagreement_ids),
            "equivalent": sum(truth),
            "not_equivalent": len(truth) - sum(truth),
        },
        "inter_rater": {
            "cohen_kappa": kappa,
            "minimum_kappa_gate": 0.6,
            "passed": kappa is not None and kappa >= 0.6,
        },
        "repeat_reliability": repeat_reliability,
        "selection_rule": "maximize_balanced_accuracy_ties_choose_higher_threshold",
        "selected_threshold": selected,
        "threshold_metrics": threshold_metrics,
        "claim_boundary": (
            "The selected boundary estimates pairwise semantic equivalence for this task and "
            "embedding model. It does not measure global scientific novelty."
        ),
    }
