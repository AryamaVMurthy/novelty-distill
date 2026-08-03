from pathlib import Path

import yaml

from novelty_distill.evaluation.research_taste import (
    METHOD_PARADIGMS,
    OPPORTUNITY_PATTERNS,
)
from novelty_distill.generation.sglang import GenerationSpec

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> GenerationSpec:
    path = ROOT / "configs" / "generation" / name
    return GenerationSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_evaluation_configs_fix_identical_sampling_across_models() -> None:
    student = _load("eval_qwen3_4b.yaml")
    adapter = _load("eval_qwen3_4b_lora.yaml")
    teacher = _load("eval_qwen3_14b.yaml")

    controls = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_new_tokens",
        "samples_per_prompt",
        "seed",
        "enable_thinking",
        "response_instruction",
    )
    assert student.samples_per_prompt == 16
    assert student.lora_path is None
    assert adapter.lora_path == "student-adapter"
    assert {
        tuple(getattr(spec, field) for field in controls) for spec in (student, adapter, teacher)
    } == {tuple(getattr(student, field) for field in controls)}


def test_production_teacher_uses_calibrated_concise_official_decoding() -> None:
    production = _load("teacher.yaml")
    evaluation = _load("eval_qwen3_14b.yaml")

    controls = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_new_tokens",
        "response_instruction",
    )
    assert tuple(getattr(production, field) for field in controls) == tuple(
        getattr(evaluation, field) for field in controls
    )
    assert production.temperature == 0.7
    assert production.top_p == 0.8
    assert production.samples_per_prompt == 8
    assert production.enable_thinking is False


def test_replication_teacher_changes_only_the_pinned_model_identity() -> None:
    main = _load("teacher.yaml")
    replication = _load("teacher_qwen3_8b.yaml")

    assert replication.model == "Qwen/Qwen3-8B"
    assert replication.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert {
        key: value
        for key, value in replication.model_dump(mode="json").items()
        if key not in {"model", "revision"}
    } == {
        key: value
        for key, value in main.model_dump(mode="json").items()
        if key not in {"model", "revision"}
    }


def test_replication_student_configs_change_only_model_identity() -> None:
    main = _load("eval_qwen3_4b.yaml")
    main_adapter = _load("eval_qwen3_4b_lora.yaml")
    replication = _load("eval_qwen3_1p7b.yaml")
    replication_adapter = _load("eval_qwen3_1p7b_lora.yaml")

    assert replication.model == "Qwen/Qwen3-1.7B"
    assert replication.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert replication_adapter.model == replication.model
    assert replication_adapter.revision == replication.revision
    assert replication.lora_path is None
    assert replication_adapter.lora_path == "student-adapter"
    ignored = {"model", "revision", "lora_path"}
    assert {
        key: value
        for key, value in replication.model_dump(mode="json").items()
        if key not in ignored
    } == {
        key: value
        for key, value in main.model_dump(mode="json").items()
        if key not in ignored
    }
    assert {
        key: value
        for key, value in replication_adapter.model_dump(mode="json").items()
        if key not in ignored
    } == {
        key: value
        for key, value in main_adapter.model_dump(mode="json").items()
        if key not in ignored
    }


def test_contrast_config_keeps_rubric_axes_descriptive() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/evaluation/primary_contrasts.yaml").read_text(encoding="utf-8")
    )

    assert set(config["metrics"]) < set(config["descriptive_metrics"])
    assert {
        "student_relevance_mean",
        "student_soundness_mean",
        "student_clarity_mean",
        "student_instruction_compliance_mean",
        "completion_tokens_mean",
    } <= set(config["descriptive_metrics"])
    assert "student_instruction_compliance_mean" not in config["metrics"]


def test_primary_teacher_cluster_threshold_is_nondegenerate_before_student_evaluation() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/evaluation/teacher_annotation.yaml").read_text(encoding="utf-8")
    )

    assert config["cosine_threshold"] == 0.94
    assert config["cosine_threshold"] in config["cosine_thresholds"]
    assert max(config["cosine_thresholds"]) == 0.95
    assert config["clustering_linkage"] == "complete"


def test_research_taste_is_attributed_frozen_and_secondary() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/evaluation/research_taste.yaml").read_text(encoding="utf-8")
    )

    assert config["status"] == "secondary_descriptive"
    assert config["annotator_max_tokens"] >= 512
    assert config["source"]["arxiv"] == "2607.01233"
    assert config["source"]["publication_status"] == "preprint"
    assert tuple(config["opportunity_patterns"]) == OPPORTUNITY_PATTERNS
    assert tuple(config["method_paradigms"]) == METHOD_PARADIGMS
    assert config["annotator_model"] == "Qwen/Qwen3-32B-FP8"
    assert config["enable_thinking"] is False
    assert config["bootstrap"] == {
        "resamples": 10000,
        "seed": 17,
        "confidence_level": 0.95,
    }
    assert config["human_validation"] == {
        "required_for_headline_claims": True,
        "minimum_records": 150,
        "minimum_annotators": 2,
        "minimum_cohen_kappa": 0.80,
    }
