import hashlib
import json
import math

import pytest

from novelty_distill.evaluation.research_taste import (
    METHOD_PARADIGMS,
    OPPORTUNITY_PATTERNS,
    ResearchTasteAnnotation,
    ResearchTasteSpec,
    analyze_research_taste_matrix,
    bootstrap_research_taste_gap,
    build_research_taste_payload,
    compare_label_distributions,
    compare_research_taste_records,
    parse_research_taste_response,
    render_research_taste_markdown,
    research_taste_protocol_hash,
    summarize_research_taste,
)
from novelty_distill.evaluation.taste_shards import (
    load_research_taste_shard,
    validate_research_taste_shard,
)


def _spec() -> ResearchTasteSpec:
    return ResearchTasteSpec(
        model="Qwen/Qwen3-32B-FP8",
        revision="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
    )


def _response(
    *,
    opportunity: str = "explanation_gap",
    method: str = "formal_conceptual_derivation",
) -> dict[str, object]:
    return {
        "id": "chatcmpl-taste-1",
        "model": "Qwen/Qwen3-32B-FP8",
        "choices": [
            {
                "message": {
                    "content": (
                        "{"
                        f'"opportunity_pattern":"{opportunity}",'
                        f'"method_paradigm":"{method}",'
                        '"surface_stitching":false,'
                        '"surface_stitching_score":0,'
                        '"bottleneck_specificity":3,'
                        '"boilerplate_score":0'
                        "}"
                    )
                }
            }
        ],
    }


def test_research_taste_payload_is_deterministic_and_strict() -> None:
    payload = build_research_taste_payload(
        prompt="Develop a hypothesis about robust catalysts.",
        response="A missing mechanistic account can be tested with a formal kinetic model.",
        spec=_spec(),
    )

    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payload
    assert "regex" not in payload
    schema = ResearchTasteAnnotation.model_json_schema()
    assert set(schema["required"]) == {
        "opportunity_pattern",
        "method_paradigm",
        "surface_stitching",
        "surface_stitching_score",
        "bottleneck_specificity",
        "boilerplate_score",
    }
    system_prompt = payload["messages"][0]["content"]
    assert "problem-finding" in system_prompt
    assert "Do not classify by scientific topic" in system_prompt
    assert "Compare all categories before deciding" in system_prompt
    assert "Return exactly one JSON object" in system_prompt
    assert "Proposal motivation and method:" in payload["messages"][1]["content"]
    assert set(schema["properties"]["opportunity_pattern"]["enum"]) == set(
        OPPORTUNITY_PATTERNS
    )
    assert set(schema["properties"]["method_paradigm"]["enum"]) == set(
        METHOD_PARADIGMS
    )


def test_research_taste_payload_names_every_exact_output_key() -> None:
    payload = build_research_taste_payload(
        prompt="Develop a hypothesis about robust catalysts.",
        response="A missing mechanistic account can be tested with a formal kinetic model.",
        spec=_spec(),
    )
    system_prompt = payload["messages"][0]["content"]

    for key in ResearchTasteAnnotation.model_fields:
        assert f'"{key}"' in system_prompt


def test_research_taste_response_validates_labels_and_diagnostics() -> None:
    result = parse_research_taste_response(_response(), _spec())

    assert result.opportunity_pattern == "explanation_gap"
    assert result.method_paradigm == "formal_conceptual_derivation"
    assert result.bottleneck_specificity == 3
    assert result.request_id == "chatcmpl-taste-1"


def test_research_taste_response_accepts_one_json_markdown_fence() -> None:
    response = _response()
    content = response["choices"][0]["message"]["content"]
    response["choices"][0]["message"]["content"] = f"```json\n{content}\n```"

    result = parse_research_taste_response(response, _spec())

    assert result.opportunity_pattern == "explanation_gap"
    assert result.bottleneck_specificity == 3


def test_research_taste_response_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="invalid structured research-taste annotation"):
        parse_research_taste_response(_response(opportunity="generic_novelty"), _spec())


def test_distribution_comparison_matches_known_binary_case() -> None:
    metrics = compare_label_distributions(
        candidate=("a", "a", "a", "a"),
        reference=("a", "a", "b", "b"),
        categories=("a", "b"),
    )

    assert metrics["candidate_normalized_entropy"] == 0.0
    assert metrics["reference_normalized_entropy"] == 1.0
    assert metrics["total_variation_distance"] == pytest.approx(0.5)
    assert metrics["jensen_shannon_divergence"] == pytest.approx(0.31127812445913283)
    assert metrics["candidate_shares"] == {"a": 1.0, "b": 0.0}
    assert metrics["reference_shares"] == {"a": 0.5, "b": 0.5}


def test_distribution_comparison_rejects_labels_outside_frozen_taxonomy() -> None:
    with pytest.raises(ValueError, match="outside the frozen taxonomy"):
        compare_label_distributions(
            candidate=("a", "other"),
            reference=("a", "b"),
            categories=("a", "b"),
        )


def test_research_taste_summary_reports_k_sample_category_coverage() -> None:
    records = (
        {
            "prompt_id": "p1",
            "sample_index": 0,
            "opportunity_pattern": "explanation_gap",
            "method_paradigm": "formal_conceptual_derivation",
            "surface_stitching": False,
            "surface_stitching_score": 0,
            "bottleneck_specificity": 3,
            "boilerplate_score": 0,
        },
        {
            "prompt_id": "p1",
            "sample_index": 1,
            "opportunity_pattern": "failure_or_risk_gap",
            "method_paradigm": "failure_mitigation_or_robustification",
            "surface_stitching": False,
            "surface_stitching_score": 1,
            "bottleneck_specificity": 2,
            "boilerplate_score": 1,
        },
        {
            "prompt_id": "p2",
            "sample_index": 0,
            "opportunity_pattern": "fragmentation_or_bridge_opportunity",
            "method_paradigm": "explicit_synthesis_or_unification",
            "surface_stitching": True,
            "surface_stitching_score": 3,
            "bottleneck_specificity": 1,
            "boilerplate_score": 2,
        },
        {
            "prompt_id": "p2",
            "sample_index": 1,
            "opportunity_pattern": "fragmentation_or_bridge_opportunity",
            "method_paradigm": "explicit_synthesis_or_unification",
            "surface_stitching": True,
            "surface_stitching_score": 2,
            "bottleneck_specificity": 1,
            "boilerplate_score": 3,
        },
    )

    summary = summarize_research_taste(records)

    assert summary["num_prompts"] == 2
    assert summary["samples_per_prompt"] == [2]
    assert summary["opportunity_category_coverage_mean"] == pytest.approx(1.5)
    assert summary["method_category_coverage_mean"] == pytest.approx(1.5)
    assert summary["joint_category_coverage_mean"] == pytest.approx(1.5)
    assert summary["opportunity_within_prompt_normalized_entropy_mean"] == pytest.approx(
        0.5 / math.log2(7)
    )
    assert summary["method_within_prompt_normalized_entropy_mean"] == pytest.approx(
        0.5 / math.log2(7)
    )
    assert summary["opportunity_prompt_unanimity_rate"] == 0.5
    assert summary["method_prompt_unanimity_rate"] == 0.5
    assert summary["bridge_opportunity_rate"] == 0.5
    assert summary["synthesis_method_rate"] == 0.5
    assert summary["surface_stitching_rate"] == 0.5
    assert summary["bottleneck_specificity_mean"] == pytest.approx(1.75)


def test_research_taste_summary_rejects_duplicate_prompt_sample_keys() -> None:
    record = {
        "prompt_id": "p1",
        "sample_index": 0,
        "opportunity_pattern": "explanation_gap",
        "method_paradigm": "formal_conceptual_derivation",
        "surface_stitching": False,
        "surface_stitching_score": 0,
        "bottleneck_specificity": 3,
        "boilerplate_score": 0,
    }

    with pytest.raises(ValueError, match="duplicate prompt/sample"):
        summarize_research_taste((record, record))


def test_research_taste_markdown_surfaces_prompt_conditioning_diagnostics() -> None:
    summary = {
        "opportunity_normalized_entropy": 0.5,
        "method_normalized_entropy": 0.6,
        "opportunity_within_prompt_normalized_entropy_mean": 0.1,
        "method_within_prompt_normalized_entropy_mean": 0.2,
        "opportunity_prompt_unanimity_rate": 0.9,
        "method_prompt_unanimity_rate": 0.8,
        "bridge_opportunity_rate": 0.3,
        "synthesis_method_rate": 0.4,
    }
    zero_jsd = {"jensen_shannon_divergence": 0.0}
    result = {
        "methods": {
            "A3": {
                "summary_all_samples": summary,
                "all_samples": {
                    "vs_human": {
                        "opportunity_pattern": zero_jsd,
                        "method_paradigm": zero_jsd,
                    }
                },
            }
        }
    }

    report = render_research_taste_markdown(
        result, {"status": "secondary_descriptive"}
    )

    assert "Opp. within-prompt H" in report
    assert "| A3 | 0.0000 | 0.0000 | 0.5000 | 0.6000 | 0.1000 |" in report
    assert "A3 has K=1" in report


def test_research_taste_comparison_requires_the_same_prompt_population() -> None:
    candidate = (
        {
            "prompt_id": "p1",
            "sample_index": 0,
            "opportunity_pattern": "fragmentation_or_bridge_opportunity",
            "method_paradigm": "explicit_synthesis_or_unification",
            "surface_stitching": True,
            "surface_stitching_score": 2,
            "bottleneck_specificity": 1,
            "boilerplate_score": 2,
        },
    )
    reference = ({**candidate[0], "prompt_id": "p2"},)

    with pytest.raises(ValueError, match="prompt population"):
        compare_research_taste_records(candidate=candidate, reference=reference)


def test_research_taste_comparison_reports_both_taxonomy_axes() -> None:
    reference = (
        {
            "prompt_id": f"p{index}",
            "sample_index": 0,
            "opportunity_pattern": label,
            "method_paradigm": method,
            "surface_stitching": False,
            "surface_stitching_score": 0,
            "bottleneck_specificity": 3,
            "boilerplate_score": 0,
        }
        for index, (label, method) in enumerate(
            (
                ("explanation_gap", "formal_conceptual_derivation"),
                ("failure_or_risk_gap", "failure_mitigation_or_robustification"),
            )
        )
    )
    reference = tuple(reference)
    candidate = tuple(
        {
            **record,
            "opportunity_pattern": "fragmentation_or_bridge_opportunity",
            "method_paradigm": "explicit_synthesis_or_unification",
            "surface_stitching": True,
        }
        for record in reference
    )

    comparison = compare_research_taste_records(
        candidate=candidate, reference=reference
    )

    assert comparison["opportunity_pattern"]["total_variation_distance"] == 1.0
    assert comparison["method_paradigm"]["total_variation_distance"] == 1.0
    assert comparison["diagnostic_deltas"]["surface_stitching_rate"] == 1.0


def test_research_taste_matrix_preserves_all_sample_and_one_shot_views() -> None:
    def row(
        prompt_id: str,
        sample_index: int,
        opportunity: str,
        method: str,
    ) -> dict[str, object]:
        return {
            "prompt_id": prompt_id,
            "sample_index": sample_index,
            "opportunity_pattern": opportunity,
            "method_paradigm": method,
            "surface_stitching": False,
            "surface_stitching_score": 0,
            "bottleneck_specificity": 3,
            "boilerplate_score": 0,
        }

    human = (
        row("p1", 0, "explanation_gap", "formal_conceptual_derivation"),
        row("p2", 0, "failure_or_risk_gap", "failure_mitigation_or_robustification"),
    )
    teacher = tuple(
        row(
            prompt_id,
            sample_index,
            "fragmentation_or_bridge_opportunity",
            "explicit_synthesis_or_unification",
        )
        for prompt_id in ("p1", "p2")
        for sample_index in (0, 1)
    )
    candidate = tuple(
        row(
            prompt_id,
            sample_index,
            "explanation_gap",
            "formal_conceptual_derivation",
        )
        for prompt_id in ("p1", "p2")
        for sample_index in (0, 1)
    )

    result = analyze_research_taste_matrix(
        {"A3": human, "A1": teacher, "B3": candidate},
        human_method="A3",
        teacher_method="A1",
    )

    assert result["status"] == "secondary_descriptive"
    assert result["methods"]["B3"]["summary_all_samples"]["samples_per_prompt"] == [
        2
    ]
    assert result["methods"]["B3"]["summary_sample_zero"]["samples_per_prompt"] == [
        1
    ]
    assert (
        result["methods"]["B3"]["all_samples"]["human_jsd_delta_vs_teacher"]
        ["opportunity_pattern"]
        < 0
    )

    bootstrapped = analyze_research_taste_matrix(
        {"A3": human, "A1": teacher, "B3": candidate},
        human_method="A3",
        teacher_method="A1",
        bootstrap_resamples=50,
        bootstrap_seed=17,
    )
    assert (
        bootstrapped["methods"]["B3"]["all_samples"]["paired_prompt_bootstrap"]
        ["resamples"]
        == 50
    )


def test_prompt_bootstrap_is_paired_and_deterministic() -> None:
    def row(
        prompt_id: str,
        opportunity: str,
        method: str,
    ) -> dict[str, object]:
        return {
            "prompt_id": prompt_id,
            "sample_index": 0,
            "opportunity_pattern": opportunity,
            "method_paradigm": method,
            "surface_stitching": opportunity
            == "fragmentation_or_bridge_opportunity",
            "surface_stitching_score": 2,
            "bottleneck_specificity": 1,
            "boilerplate_score": 2,
        }

    human = tuple(
        row(f"p{index}", "explanation_gap", "formal_conceptual_derivation")
        for index in range(8)
    )
    teacher = tuple(
        row(
            f"p{index}",
            "fragmentation_or_bridge_opportunity",
            "explicit_synthesis_or_unification",
        )
        for index in range(8)
    )

    first = bootstrap_research_taste_gap(
        candidate=teacher,
        human=human,
        teacher=teacher,
        resamples=200,
        seed=17,
    )
    second = bootstrap_research_taste_gap(
        candidate=teacher,
        human=human,
        teacher=teacher,
        resamples=200,
        seed=17,
    )

    assert first == second
    assert first["unit"] == "prompt"
    assert first["metrics"]["opportunity_jsd_vs_human"] == {
        "estimate": 1.0,
        "ci_low": 1.0,
        "ci_high": 1.0,
    }
    assert first["metrics"]["opportunity_human_jsd_delta_vs_teacher"] == {
        "estimate": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
    }


def _taste_shard_payload(texts: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_hash": research_taste_protocol_hash(),
        "prompt_id": "p1",
        "text_hashes": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
        "annotator": _spec().model_dump(mode="json"),
        "records": [
            {
                "prompt_id": "p1",
                "sample_index": index,
                "text": text,
                "opportunity_pattern": "explanation_gap",
                "method_paradigm": "formal_conceptual_derivation",
                "surface_stitching": False,
                "surface_stitching_score": 0,
                "bottleneck_specificity": 3,
                "boilerplate_score": 0,
                "request_id": f"request-{index}",
                "model": _spec().model,
            }
            for index, text in enumerate(texts)
        ],
    }


def test_research_taste_shard_is_content_and_annotator_bound(tmp_path) -> None:
    texts = ("idea one", "idea two")
    path = tmp_path / "p1.json"
    path.write_text(json.dumps(_taste_shard_payload(texts)), encoding="utf-8")

    loaded = load_research_taste_shard(path, samples_per_prompt=2)

    assert [record["sample_index"] for record in loaded["records"]] == [0, 1]
    assert validate_research_taste_shard(
        path,
        prompt_id="p1",
        text_hashes=[hashlib.sha256(text.encode()).hexdigest() for text in texts],
        annotator=_spec(),
    )
    with pytest.raises(ValueError, match="stale or incompatible"):
        validate_research_taste_shard(
            path,
            prompt_id="p1",
            text_hashes=[
                hashlib.sha256(b"changed").hexdigest(),
                hashlib.sha256(b"idea two").hexdigest(),
            ],
            annotator=_spec(),
        )


def test_research_taste_shard_rejects_duplicate_request_ids(tmp_path) -> None:
    payload = _taste_shard_payload(("idea one", "idea two"))
    payload["records"][1]["request_id"] = "request-0"  # type: ignore[index]
    path = tmp_path / "p1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate annotator request"):
        load_research_taste_shard(path, samples_per_prompt=2)


def test_research_taste_shard_rejects_changed_taxonomy_protocol(tmp_path) -> None:
    payload = _taste_shard_payload(("idea one",))
    payload["protocol_hash"] = "0" * 64
    path = tmp_path / "p1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="protocol hash"):
        load_research_taste_shard(path, samples_per_prompt=1)
