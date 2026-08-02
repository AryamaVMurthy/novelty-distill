"""Research-taste annotation and distribution diagnostics for scientific ideas.

This is a secondary evaluation inspired by Chen, Zhao, and Cohan (2026).  It is
kept separate from the frozen semantic-mode and quality outcomes so that adding
the diagnostic cannot silently change the primary experiment.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OpportunityPattern = Literal[
    "puzzle_or_contradiction",
    "explanation_gap",
    "assumption_or_scope_mismatch",
    "measurement_evidence_gap",
    "fragmentation_or_bridge_opportunity",
    "failure_or_risk_gap",
    "resource_or_operational_constraint",
]
MethodParadigm = Literal[
    "explicit_synthesis_or_unification",
    "assumption_relaxation_or_scope_extension",
    "failure_mitigation_or_robustification",
    "formal_conceptual_derivation",
    "measurement_or_empirical_mapping",
    "constructive_artifact_or_system",
    "optimization_search_or_resource_strategy",
]

OPPORTUNITY_PATTERNS: tuple[str, ...] = (
    "puzzle_or_contradiction",
    "explanation_gap",
    "assumption_or_scope_mismatch",
    "measurement_evidence_gap",
    "fragmentation_or_bridge_opportunity",
    "failure_or_risk_gap",
    "resource_or_operational_constraint",
)
METHOD_PARADIGMS: tuple[str, ...] = (
    "explicit_synthesis_or_unification",
    "assumption_relaxation_or_scope_extension",
    "failure_mitigation_or_robustification",
    "formal_conceptual_derivation",
    "measurement_or_empirical_mapping",
    "constructive_artifact_or_system",
    "optimization_search_or_resource_strategy",
)


class ResearchTasteSpec(BaseModel):
    """Pinned annotator identity and deterministic decoding settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    revision: str = Field(min_length=40, max_length=40)
    max_tokens: int = Field(default=192, gt=0)


class ResearchTasteAnnotation(BaseModel):
    """Strict two-axis label and diagnostic record returned by the annotator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opportunity_pattern: OpportunityPattern
    method_paradigm: MethodParadigm
    surface_stitching: bool
    surface_stitching_score: int = Field(ge=0, le=3)
    bottleneck_specificity: int = Field(ge=0, le=3)
    boilerplate_score: int = Field(ge=0, le=3)


class JudgedResearchTaste(ResearchTasteAnnotation):
    """A validated annotation plus request-level provenance."""

    request_id: str = Field(min_length=1)
    model: str = Field(min_length=1)


class ResearchTasteRecord(ResearchTasteAnnotation):
    """One stored annotation aligned to a generated sample."""

    prompt_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)


_SYSTEM_PROMPT = """\
You annotate the research taste expressed by a proposed scientific idea. Classify two separate
properties: its problem-finding pattern and its high-level contribution strategy.
Do not classify by scientific topic,
field, model family, or named technical substrate. Several components do not
by themselves imply synthesis. Judge only the supplied task and candidate answer. Return only the
requested JSON object.

Opportunity pattern (why the work is needed):
- puzzle_or_contradiction: a paradox, unexplained tradeoff, or conflicting evidence.
- explanation_gap: a missing causal, mechanistic, theoretical, or explanatory account.
- assumption_or_scope_mismatch: narrow, hidden, incompatible, or unrealistic assumptions/scope.
- measurement_evidence_gap: missing measurement, benchmark, audit, diagnosis, or evidence map.
- fragmentation_or_bridge_opportunity: disconnected evidence, theories, methods, or communities.
- failure_or_risk_gap: brittleness, unreliability, bias, safety, privacy, or
  reproducibility failure.
- resource_or_operational_constraint: cost, compute, time, data, deployment, or scalability barrier.

Method paradigm (how the contribution addresses it):
- explicit_synthesis_or_unification: centrally bridge, reconcile, or unify separate lines of work.
- assumption_relaxation_or_scope_extension: work under broader, weaker, or more realistic regimes.
- failure_mitigation_or_robustification: directly reduce failure, risk, or unreliability.
- formal_conceptual_derivation: provide a model, theorem, proof, objective, taxonomy, or
  explanation.
- measurement_or_empirical_mapping: create systematic measurement, diagnostics, data, or evidence.
- constructive_artifact_or_system: make a concrete tool, system, platform, prototype, or material.
- optimization_search_or_resource_strategy: use search, tuning, selection, allocation, or
  efficiency.

Surface stitching means a superficial A-plus-B combination without a precise reason the pieces
must interact. Bottleneck specificity is high only when a concrete mechanism or limiting factor is
identified. Boilerplate is high when wording could apply to many unrelated research problems.
"""


def build_research_taste_payload(
    *, prompt: str, response: str, spec: ResearchTasteSpec
) -> dict[str, Any]:
    """Build a deterministic SGLang request with a strict annotation schema."""

    if not prompt.strip() or not response.strip():
        raise ValueError("research-taste prompt and response must both be non-empty")
    return {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Scientific task:\n{prompt}\n\nCandidate idea:\n{response}",
            },
        ],
        "temperature": 0,
        "max_tokens": spec.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "research_taste_annotation",
                "strict": True,
                "schema": ResearchTasteAnnotation.model_json_schema(),
            },
        },
    }


def parse_research_taste_response(
    response: Mapping[str, Any], spec: ResearchTasteSpec
) -> JudgedResearchTaste:
    """Validate a structured annotator response and preserve request provenance."""

    request_id = str(response.get("id", "")).strip()
    choices = response.get("choices")
    if not request_id:
        raise ValueError("research-taste response has no request id")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("research-taste response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("research-taste choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("research-taste choice has no text content")
    try:
        annotation = ResearchTasteAnnotation.model_validate(json.loads(message["content"]))
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise ValueError(
            "annotator returned invalid structured research-taste annotation"
        ) from error
    return JudgedResearchTaste(
        **annotation.model_dump(mode="json"),
        request_id=request_id,
        model=str(response.get("model", spec.model)),
    )


def _shares(labels: Sequence[str], categories: Sequence[str]) -> dict[str, float]:
    if not labels:
        raise ValueError("label distribution cannot be empty")
    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("taxonomy must contain at least two unique categories")
    unknown = sorted(set(labels) - set(categories))
    if unknown:
        raise ValueError(f"labels outside the frozen taxonomy: {unknown}")
    counts = Counter(labels)
    return {category: counts[category] / len(labels) for category in categories}


def _normalized_entropy(shares: Mapping[str, float]) -> float:
    return -sum(value * math.log2(value) for value in shares.values() if value) / math.log2(
        len(shares)
    )


def compare_label_distributions(
    *, candidate: Sequence[str], reference: Sequence[str], categories: Sequence[str]
) -> dict[str, Any]:
    """Compare empirical categorical distributions with base-2 JSD."""

    candidate_shares = _shares(candidate, categories)
    reference_shares = _shares(reference, categories)
    total_variation = 0.5 * sum(
        abs(candidate_shares[category] - reference_shares[category])
        for category in categories
    )
    midpoint = {
        category: (candidate_shares[category] + reference_shares[category]) / 2
        for category in categories
    }

    def kl(left: Mapping[str, float]) -> float:
        return sum(
            left[category] * math.log2(left[category] / midpoint[category])
            for category in categories
            if left[category]
        )

    return {
        "candidate_shares": candidate_shares,
        "reference_shares": reference_shares,
        "candidate_normalized_entropy": _normalized_entropy(candidate_shares),
        "reference_normalized_entropy": _normalized_entropy(reference_shares),
        "total_variation_distance": total_variation,
        "jensen_shannon_divergence": 0.5 * (kl(candidate_shares) + kl(reference_shares)),
    }


def summarize_research_taste(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize global taste concentration and within-prompt K-sample coverage."""

    if not records:
        raise ValueError("research-taste summary requires at least one record")
    normalized = tuple(ResearchTasteRecord.model_validate(record) for record in records)
    keys = [(record.prompt_id, record.sample_index) for record in normalized]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate prompt/sample research-taste record")

    by_prompt: dict[str, list[ResearchTasteRecord]] = defaultdict(list)
    for record in normalized:
        by_prompt[record.prompt_id].append(record)
    sample_counts = sorted({len(prompt_records) for prompt_records in by_prompt.values()})
    opportunity_labels = tuple(record.opportunity_pattern for record in normalized)
    method_labels = tuple(record.method_paradigm for record in normalized)
    opportunity_shares = _shares(opportunity_labels, OPPORTUNITY_PATTERNS)
    method_shares = _shares(method_labels, METHOD_PARADIGMS)
    opportunity_coverage = [
        len({record.opportunity_pattern for record in prompt_records})
        for prompt_records in by_prompt.values()
    ]
    method_coverage = [
        len({record.method_paradigm for record in prompt_records})
        for prompt_records in by_prompt.values()
    ]
    joint_coverage = [
        len(
            {
                (record.opportunity_pattern, record.method_paradigm)
                for record in prompt_records
            }
        )
        for prompt_records in by_prompt.values()
    ]
    return {
        "num_prompts": len(by_prompt),
        "num_records": len(normalized),
        "samples_per_prompt": sample_counts,
        "opportunity_shares": opportunity_shares,
        "method_shares": method_shares,
        "opportunity_normalized_entropy": _normalized_entropy(opportunity_shares),
        "method_normalized_entropy": _normalized_entropy(method_shares),
        "opportunity_category_coverage_mean": statistics.fmean(opportunity_coverage),
        "method_category_coverage_mean": statistics.fmean(method_coverage),
        "joint_category_coverage_mean": statistics.fmean(joint_coverage),
        "bridge_opportunity_rate": opportunity_shares[
            "fragmentation_or_bridge_opportunity"
        ],
        "synthesis_method_rate": method_shares["explicit_synthesis_or_unification"],
        "surface_stitching_rate": statistics.fmean(
            int(record.surface_stitching) for record in normalized
        ),
        "surface_stitching_score_mean": statistics.fmean(
            record.surface_stitching_score for record in normalized
        ),
        "bottleneck_specificity_mean": statistics.fmean(
            record.bottleneck_specificity for record in normalized
        ),
        "boilerplate_score_mean": statistics.fmean(
            record.boilerplate_score for record in normalized
        ),
    }


def compare_research_taste_records(
    *,
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two methods evaluated on the same prompt population."""

    candidate_records = tuple(ResearchTasteRecord.model_validate(row) for row in candidate)
    reference_records = tuple(ResearchTasteRecord.model_validate(row) for row in reference)
    candidate_prompts = {record.prompt_id for record in candidate_records}
    reference_prompts = {record.prompt_id for record in reference_records}
    if not candidate_prompts or candidate_prompts != reference_prompts:
        raise ValueError("candidate and reference must use exactly the same prompt population")
    candidate_summary = summarize_research_taste(
        tuple(record.model_dump(mode="json") for record in candidate_records)
    )
    reference_summary = summarize_research_taste(
        tuple(record.model_dump(mode="json") for record in reference_records)
    )
    diagnostic_names = (
        "surface_stitching_rate",
        "surface_stitching_score_mean",
        "bottleneck_specificity_mean",
        "boilerplate_score_mean",
    )
    return {
        "opportunity_pattern": compare_label_distributions(
            candidate=tuple(record.opportunity_pattern for record in candidate_records),
            reference=tuple(record.opportunity_pattern for record in reference_records),
            categories=OPPORTUNITY_PATTERNS,
        ),
        "method_paradigm": compare_label_distributions(
            candidate=tuple(record.method_paradigm for record in candidate_records),
            reference=tuple(record.method_paradigm for record in reference_records),
            categories=METHOD_PARADIGMS,
        ),
        "diagnostic_deltas": {
            name: candidate_summary[name] - reference_summary[name]
            for name in diagnostic_names
        },
    }


def analyze_research_taste_matrix(
    methods: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    human_method: str,
    teacher_method: str,
) -> dict[str, Any]:
    """Build all-sample and matched-one-shot secondary taste comparisons."""

    if human_method not in methods or teacher_method not in methods:
        raise ValueError("research-taste matrix requires named human and teacher methods")
    if len(methods) < 3:
        raise ValueError("research-taste matrix requires human, teacher, and a candidate")
    normalized: dict[str, tuple[dict[str, Any], ...]] = {}
    for method, records in methods.items():
        if not method.strip():
            raise ValueError("research-taste method names must be non-empty")
        validated = tuple(ResearchTasteRecord.model_validate(record) for record in records)
        summary = summarize_research_taste(
            tuple(record.model_dump(mode="json") for record in validated)
        )
        if len(summary["samples_per_prompt"]) != 1:
            raise ValueError(f"method {method!r} has unequal samples per prompt")
        normalized[method] = tuple(record.model_dump(mode="json") for record in validated)

    human_records = normalized[human_method]
    teacher_records = normalized[teacher_method]
    teacher_human_all = compare_research_taste_records(
        candidate=teacher_records, reference=human_records
    )

    def sample_zero(records: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
        selected = tuple(record for record in records if record["sample_index"] == 0)
        if {record["prompt_id"] for record in selected} != {
            record["prompt_id"] for record in records
        }:
            raise ValueError("every prompt must contain deterministic sample index zero")
        return selected

    human_zero = sample_zero(human_records)
    teacher_zero = sample_zero(teacher_records)
    teacher_human_zero = compare_research_taste_records(
        candidate=teacher_zero, reference=human_zero
    )
    analyses: dict[str, Any] = {}
    for method in sorted(normalized):
        records = normalized[method]
        zero_records = sample_zero(records)
        all_vs_human = compare_research_taste_records(
            candidate=records, reference=human_records
        )
        all_vs_teacher = compare_research_taste_records(
            candidate=records, reference=teacher_records
        )
        zero_vs_human = compare_research_taste_records(
            candidate=zero_records, reference=human_zero
        )
        zero_vs_teacher = compare_research_taste_records(
            candidate=zero_records, reference=teacher_zero
        )
        analyses[method] = {
            "summary_all_samples": summarize_research_taste(records),
            "summary_sample_zero": summarize_research_taste(zero_records),
            "all_samples": {
                "vs_human": all_vs_human,
                "vs_teacher": all_vs_teacher,
                "human_jsd_delta_vs_teacher": {
                    axis: all_vs_human[axis]["jensen_shannon_divergence"]
                    - teacher_human_all[axis]["jensen_shannon_divergence"]
                    for axis in ("opportunity_pattern", "method_paradigm")
                },
            },
            "sample_zero": {
                "vs_human": zero_vs_human,
                "vs_teacher": zero_vs_teacher,
                "human_jsd_delta_vs_teacher": {
                    axis: zero_vs_human[axis]["jensen_shannon_divergence"]
                    - teacher_human_zero[axis]["jensen_shannon_divergence"]
                    for axis in ("opportunity_pattern", "method_paradigm")
                },
            },
        }
    return {
        "schema_version": 1,
        "status": "secondary_descriptive",
        "human_method": human_method,
        "teacher_method": teacher_method,
        "methods": analyses,
    }
