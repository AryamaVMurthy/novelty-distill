from pathlib import Path

import yaml

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
