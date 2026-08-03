from pathlib import Path

import pytest

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.opsd import (
    OPSDRunSpec,
    build_opsd_rows,
    load_opsd_run_spec,
    opsd_dataset_kwargs,
    override_opsd_baseline,
    render_matched_prompt_pairs,
    render_privileged_prompt_pairs,
)


def test_privileged_opsd_smoke_is_pinned_and_uses_official_row_schema() -> None:
    spec = load_opsd_run_spec(Path("configs/training/opsd_smoke.yaml"))
    registry = load_baseline_registry(Path("configs/baselines.yaml"))
    baseline = next(item for item in registry.baselines if item.id == spec.baseline_id)
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Can a coating improve stability?",
            "background_survey": "Humidity damages the catalyst.",
            "fine_grained_hypothesis": "A hydrophobic coating will help.",
            "inspiration": [{"insp": "A privileged direction."}],
        },
        split="train",
        task="open",
    )

    rows = build_opsd_rows(baseline, (example,))

    assert spec.model == "Qwen/Qwen3-1.7B"
    assert spec.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert spec.max_steps == 1
    assert rows[0]["problem"] == example.student_prompt
    assert "A hydrophobic coating" in rows[0]["solution"]
    assert "A privileged direction" in rows[0]["solution"]


def test_no_privilege_control_removes_all_teacher_only_information() -> None:
    registry = load_baseline_registry(Path("configs/baselines.yaml"))
    baseline = next(item for item in registry.baselines if item.id == "E4")
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Can a coating improve stability?",
            "background_survey": "Humidity damages the catalyst.",
            "fine_grained_hypothesis": "A hydrophobic coating will help.",
            "inspiration": [{"insp": "A privileged direction."}],
        },
        split="train",
        task="open",
    )

    rows = build_opsd_rows(baseline, (example,))

    assert rows == ({"id": "paper-1", "problem": example.student_prompt, "solution": ""},)


def test_matched_context_control_renders_identical_teacher_and_student_prompts() -> None:
    class RecordingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return f"rendered::{messages[0]['content']}"

    pairs = render_matched_prompt_pairs(
        ({"problem": "ordinary prompt", "solution": "must be ignored"},),
        tokenizer=RecordingTokenizer(),
        enable_thinking=False,
    )

    assert pairs == (("rendered::ordinary prompt", "rendered::ordinary prompt"),)


def test_privileged_collator_changes_only_the_teacher_context_not_the_task() -> None:
    class RecordingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return f"rendered::{messages[0]['content']}"

    features = ({"problem": "ordinary prompt", "solution": "private evidence"},)
    student, teacher = render_privileged_prompt_pairs(
        features,
        tokenizer=RecordingTokenizer(),
        student_thinking=False,
        teacher_thinking=False,
    )[0]
    matched_student = render_matched_prompt_pairs(
        features,
        tokenizer=RecordingTokenizer(),
        enable_thinking=False,
    )[0][0]

    assert student == matched_student
    assert "private evidence" not in student
    assert "ordinary prompt" in teacher
    assert "private evidence" in teacher
    assert "boxed" not in teacher.lower()
    assert "step by step" not in teacher.lower()


def test_official_opsd_receives_raw_problem_solution_rows() -> None:
    assert opsd_dataset_kwargs() == {"skip_prepare_dataset": True}


def test_opsd_smoke_config_can_select_each_supported_ablation() -> None:
    base = load_opsd_run_spec(Path("configs/training/opsd_smoke.yaml"))

    assert override_opsd_baseline(base, "E3").output_dir == Path("checkpoints/E3-smoke")
    assert override_opsd_baseline(base, "E4").baseline_id == "E4"
    assert override_opsd_baseline(
        base, "E2", run_suffix="tomato1k-seed17"
    ).output_dir == Path("checkpoints/E2-tomato1k-seed17")


def test_opsd_research_run_checkpoints_before_the_six_hour_boundary() -> None:
    spec = load_opsd_run_spec(Path("configs/training/opsd_tomato1k.yaml"))

    assert spec.save_steps == 25


def test_opsd_runs_do_not_clip_signed_vocabulary_contributions() -> None:
    for path in sorted(Path("configs/training").glob("opsd_*.yaml")):
        spec = load_opsd_run_spec(path)
        assert spec.jsd_token_clip is None, path


def test_opsd_run_spec_rejects_pre_vocabulary_divergence_clipping() -> None:
    payload = load_opsd_run_spec(
        Path("configs/training/opsd_tomato1k.yaml")
    ).model_dump()
    payload["jsd_token_clip"] = 0.05

    with pytest.raises(ValueError, match="signed vocabulary contributions"):
        OPSDRunSpec.model_validate(payload)
