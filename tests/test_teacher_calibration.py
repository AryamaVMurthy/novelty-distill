from pathlib import Path

import pytest
import yaml

from novelty_distill.evaluation.teacher_calibration import (
    TeacherCalibrationStudy,
    select_length_stratified,
)


def test_length_stratification_is_stable_and_spans_prompt_lengths() -> None:
    rows = tuple(
        {"id": f"p-{index:02d}", "student_prompt": "x" * (index + 1)}
        for index in range(16)
    )

    selected = select_length_stratified(rows, size=4)
    selected_reversed = select_length_stratified(reversed(rows), size=4)

    assert [row["id"] for row in selected] == ["p-01", "p-05", "p-09", "p-13"]
    assert [row["id"] for row in selected_reversed] == [
        "p-01",
        "p-05",
        "p-09",
        "p-13",
    ]


def test_length_stratification_rejects_invalid_or_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="at least two"):
        select_length_stratified(({"id": "p", "student_prompt": "x"},), size=1)
    with pytest.raises(ValueError, match="unique"):
        select_length_stratified(
            (
                {"id": "p", "student_prompt": "x"},
                {"id": "p", "student_prompt": "xx"},
            ),
            size=2,
        )


def test_teacher_calibration_config_has_frozen_distinct_conditions() -> None:
    payload = yaml.safe_load(
        Path("configs/generation/teacher_calibration.yaml").read_text(encoding="utf-8")
    )

    study = TeacherCalibrationStudy.model_validate(payload)

    assert study.prompt_count == 8
    assert study.concurrency == 8
    assert len(study.conditions) == 7
    assert len({condition.id for condition in study.conditions}) == 7
    assert {condition.generation.temperature for condition in study.conditions} == {
        0.6,
        0.7,
        0.8,
        1.0,
        1.2,
    }
    assert all(not condition.generation.enable_thinking for condition in study.conditions)
    assert all(condition.generation.top_k == 20 for condition in study.conditions)
    assert all(condition.generation.min_p == 0 for condition in study.conditions)


def test_concision_calibration_changes_only_the_response_requirement() -> None:
    payload = yaml.safe_load(
        Path("configs/generation/teacher_concision_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )

    study = TeacherCalibrationStudy.model_validate(payload)

    assert study.reference_condition == "t07-p08-l512"
    assert [condition.id for condition in study.conditions] == [
        "t07-p08-l512",
        "concise-300-words",
    ]
    reference, concise = (condition.generation for condition in study.conditions)
    assert reference.response_instruction is None
    assert concise.response_instruction is not None
    assert concise.model_dump(exclude={"response_instruction"}) == reference.model_dump(
        exclude={"response_instruction"}
    )


def test_calibration_rejects_a_reference_condition_outside_the_matrix() -> None:
    payload = yaml.safe_load(
        Path("configs/generation/teacher_concision_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["reference_condition"] = "missing-condition"

    with pytest.raises(ValueError, match="reference condition"):
        TeacherCalibrationStudy.model_validate(payload)


def test_concise_temperature_followup_keeps_the_length_intervention_fixed() -> None:
    payload = yaml.safe_load(
        Path("configs/generation/teacher_concise_temperature_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )

    study = TeacherCalibrationStudy.model_validate(payload)

    assert study.reference_condition == "concise-t07"
    assert {condition.generation.temperature for condition in study.conditions} == {
        0.7,
        0.8,
        1.0,
        1.2,
    }
    assert len({condition.generation.response_instruction for condition in study.conditions}) == 1
