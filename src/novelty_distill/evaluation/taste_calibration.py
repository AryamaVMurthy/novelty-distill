"""Blinded human calibration and agreement gates for research-taste labels."""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from novelty_distill.evaluation.research_taste import (
    METHOD_PARADIGMS,
    OPPORTUNITY_PATTERNS,
    MethodParadigm,
    OpportunityPattern,
)


class TasteCalibrationCandidate(BaseModel):
    """One auto-annotated sample eligible for blinded human calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    task: str = Field(min_length=1)
    text: str = Field(min_length=1)
    opportunity_pattern: OpportunityPattern
    method_paradigm: MethodParadigm


class HumanTasteLabel(BaseModel):
    """The only fields accepted from one blinded human annotation file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calibration_id: str = Field(min_length=1)
    opportunity_pattern: OpportunityPattern
    method_paradigm: MethodParadigm


def normalize_human_packet_row(row: Mapping[str, Any]) -> dict[str, str]:
    """Accept an edited blinded packet row while rejecting unrecognized fields."""

    allowed = {
        "calibration_id",
        "task",
        "candidate_idea",
        "opportunity_pattern",
        "method_paradigm",
    }
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise ValueError(f"unexpected human label fields: {unknown}")
    label = HumanTasteLabel.model_validate(
        {
            key: row.get(key)
            for key in (
                "calibration_id",
                "opportunity_pattern",
                "method_paradigm",
            )
        }
    )
    return label.model_dump(mode="json")


_STRATA = ("human", "teacher", "base", "trained")


def _stratum(method: str) -> str:
    return {"A3": "human", "A1": "teacher", "A0": "base"}.get(method, "trained")


def _selection_digest(candidate: TasteCalibrationCandidate, *, seed: int) -> str:
    encoded = (
        f"{seed}\0{candidate.method}\0{candidate.prompt_id}\0"
        f"{candidate.sample_index}\0{candidate.text}"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _select_round_robin(
    candidates: Sequence[TasteCalibrationCandidate], *, size: int, seed: int
) -> tuple[TasteCalibrationCandidate, ...]:
    by_method: dict[str, list[TasteCalibrationCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_method[candidate.method].append(candidate)
    for method in by_method:
        by_method[method].sort(key=lambda candidate: _selection_digest(candidate, seed=seed))
    method_order = sorted(
        by_method,
        key=lambda method: hashlib.sha256(f"{seed}\0{method}".encode()).hexdigest(),
    )
    selected: list[TasteCalibrationCandidate] = []
    offset = 0
    while len(selected) < size:
        progressed = False
        for method in method_order:
            if offset < len(by_method[method]):
                selected.append(by_method[method][offset])
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            break
        offset += 1
    return tuple(selected)


def select_blinded_taste_calibration(
    candidates: Sequence[Mapping[str, Any]], *, size: int, seed: int
) -> dict[str, Any]:
    """Select a balanced packet and a separate identity/automatic-label key."""

    if size < len(_STRATA):
        raise ValueError("calibration size must cover all four source strata")
    validated = tuple(TasteCalibrationCandidate.model_validate(row) for row in candidates)
    identities = [
        (candidate.method, candidate.prompt_id, candidate.sample_index) for candidate in validated
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate method/prompt/sample calibration candidate")
    grouped: dict[str, list[TasteCalibrationCandidate]] = defaultdict(list)
    for candidate in validated:
        grouped[_stratum(candidate.method)].append(candidate)
    quotient, remainder = divmod(size, len(_STRATA))
    quotas = {
        stratum: quotient + int(index < remainder)
        for index, stratum in enumerate(_STRATA)
    }
    selected: list[tuple[str, TasteCalibrationCandidate]] = []
    for stratum in _STRATA:
        available = grouped[stratum]
        if len(available) < quotas[stratum]:
            raise ValueError(
                f"{stratum} calibration stratum has {len(available)} records, "
                f"requires {quotas[stratum]}"
            )
        chosen = _select_round_robin(available, size=quotas[stratum], seed=seed)
        if len(chosen) != quotas[stratum]:
            raise ValueError(f"could not fill {stratum} calibration stratum")
        selected.extend((stratum, candidate) for candidate in chosen)

    packet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for stratum, candidate in selected:
        calibration_id = hashlib.sha256(
            (
                f"taste-calibration-v1\0{seed}\0{candidate.method}\0"
                f"{candidate.prompt_id}\0{candidate.sample_index}\0{candidate.text}"
            ).encode()
        ).hexdigest()[:24]
        packet.append(
            {
                "calibration_id": calibration_id,
                "task": candidate.task,
                "candidate_idea": candidate.text,
                "opportunity_pattern": None,
                "method_paradigm": None,
            }
        )
        key.append(
            {
                "calibration_id": calibration_id,
                "stratum": stratum,
                "method": candidate.method,
                "prompt_id": candidate.prompt_id,
                "sample_index": candidate.sample_index,
                "text_sha256": hashlib.sha256(candidate.text.encode()).hexdigest(),
                "opportunity_pattern": candidate.opportunity_pattern,
                "method_paradigm": candidate.method_paradigm,
            }
        )
    order = sorted(
        range(len(packet)),
        key=lambda index: hashlib.sha256(
            f"{seed}\0blind\0{packet[index]['calibration_id']}".encode()
        ).hexdigest(),
    )
    return {
        "schema_version": 1,
        "size": size,
        "seed": seed,
        "stratum_counts": quotas,
        "packet": tuple(packet[index] for index in order),
        "key": tuple(sorted(key, key=lambda record: record["calibration_id"])),
    }


def cohen_kappa(
    left: Sequence[str], right: Sequence[str], *, categories: Sequence[str]
) -> float:
    """Compute unweighted Cohen's kappa for one frozen categorical axis."""

    if not left or len(left) != len(right):
        raise ValueError("Cohen kappa requires equal non-empty label sequences")
    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("Cohen kappa categories must be unique and non-degenerate")
    unknown = sorted((set(left) | set(right)) - set(categories))
    if unknown:
        raise ValueError(f"Cohen kappa labels outside taxonomy: {unknown}")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[category] / len(left)) * (right_counts[category] / len(right))
        for category in categories
    )
    if expected == 1:
        raise ValueError("Cohen kappa is undefined for a single observed category")
    return (observed - expected) / (1 - expected)


def analyze_human_taste_agreement(
    *,
    key: Sequence[Mapping[str, Any]],
    humans: Mapping[str, Sequence[Mapping[str, Any]]],
    minimum_records: int,
    minimum_annotators: int,
    minimum_kappa: float,
) -> dict[str, Any]:
    """Validate complete blinded labels and fail closed unless every kappa gate passes."""

    if len(key) < minimum_records:
        raise ValueError(
            f"human calibration has {len(key)} records, requires {minimum_records}"
        )
    if len(humans) < minimum_annotators:
        raise ValueError(
            f"human calibration has {len(humans)} annotators, requires {minimum_annotators}"
        )
    if not -1 <= minimum_kappa <= 1:
        raise ValueError("minimum Cohen kappa must be between -1 and 1")
    key_ids = [str(record.get("calibration_id", "")) for record in key]
    if any(not calibration_id for calibration_id in key_ids) or len(set(key_ids)) != len(key_ids):
        raise ValueError("calibration key IDs must be non-empty and unique")
    automated = {
        str(record["calibration_id"]): HumanTasteLabel.model_validate(
            {
                "calibration_id": record["calibration_id"],
                "opportunity_pattern": record["opportunity_pattern"],
                "method_paradigm": record["method_paradigm"],
            }
        )
        for record in key
    }
    normalized_humans: dict[str, dict[str, HumanTasteLabel]] = {}
    for annotator, rows in humans.items():
        if not annotator.strip() or annotator in normalized_humans:
            raise ValueError("human annotator names must be unique and non-empty")
        labels = tuple(HumanTasteLabel.model_validate(row) for row in rows)
        by_id = {label.calibration_id: label for label in labels}
        if len(by_id) != len(labels) or set(by_id) != set(key_ids):
            raise ValueError(
                f"annotator {annotator!r} must provide the exact calibration IDs"
            )
        normalized_humans[annotator] = by_id

    axes = {
        "opportunity_pattern": OPPORTUNITY_PATTERNS,
        "method_paradigm": METHOD_PARADIGMS,
    }
    automated_vs_human: dict[str, dict[str, float]] = {axis: {} for axis in axes}
    human_vs_human: dict[str, dict[str, float]] = {axis: {} for axis in axes}
    failures: list[str] = []
    for axis, categories in axes.items():
        automated_labels = tuple(
            str(getattr(automated[calibration_id], axis)) for calibration_id in key_ids
        )
        for annotator in sorted(normalized_humans):
            human_labels = tuple(
                str(getattr(normalized_humans[annotator][calibration_id], axis))
                for calibration_id in key_ids
            )
            value = cohen_kappa(automated_labels, human_labels, categories=categories)
            automated_vs_human[axis][annotator] = value
            if value < minimum_kappa:
                failures.append(f"automated_vs_human.{axis}.{annotator}")
        for left, right in itertools.combinations(sorted(normalized_humans), 2):
            pair_name = f"{left}__{right}"
            value = cohen_kappa(
                tuple(
                    str(getattr(normalized_humans[left][calibration_id], axis))
                    for calibration_id in key_ids
                ),
                tuple(
                    str(getattr(normalized_humans[right][calibration_id], axis))
                    for calibration_id in key_ids
                ),
                categories=categories,
            )
            human_vs_human[axis][pair_name] = value
            if value < minimum_kappa:
                failures.append(f"human_vs_human.{axis}.{pair_name}")
    return {
        "schema_version": 1,
        "passed": not failures,
        "num_records": len(key_ids),
        "annotators": sorted(normalized_humans),
        "minimum_kappa": minimum_kappa,
        "automated_vs_human": automated_vs_human,
        "human_vs_human": human_vs_human,
        "failures": failures,
    }
