import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from novelty_distill.data.coverage_residual import (
    CoverageCandidate,
    CoverageResidualParameters,
    StudentProbe,
    build_coverage_residual_artifact,
    residual_weights,
    select_residual_target,
    vmf_similarity_density,
)


def test_vmf_density_is_finite_for_duplicate_normalized_points() -> None:
    teacher = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    student = np.asarray([[1.0, 0.0], [1.0, 0.0]])

    density = vmf_similarity_density(teacher, student, kappa=8.0)

    assert density[0] == pytest.approx(1.0)
    assert density[1] == pytest.approx(math.exp(-8.0))
    assert np.isfinite(density).all()


def test_vmf_density_rejects_non_normalized_or_dimension_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="normalized"):
        vmf_similarity_density(np.asarray([[2.0, 0.0]]), np.asarray([[1.0, 0.0]]), kappa=8.0)
    with pytest.raises(ValueError, match="dimension"):
        vmf_similarity_density(np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0, 0.0]]), kappa=8.0)


def test_invalid_outlier_receives_zero_residual_weight() -> None:
    weights = residual_weights(
        np.asarray([0.95, 0.40]),
        np.asarray([0.9, 0.0001]),
        minimum_quality=0.6,
        gamma=1.0,
        epsilon=1e-3,
    )

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == 0.0


def test_undercovered_candidate_wins_at_equal_quality_and_gamma_controls_effect() -> None:
    quality = np.asarray([0.8, 0.8])
    density = np.asarray([0.8, 0.2])

    quality_only = residual_weights(quality, density, minimum_quality=0.6, gamma=0.0, epsilon=1e-3)
    residual = residual_weights(quality, density, minimum_quality=0.6, gamma=1.0, epsilon=1e-3)

    assert quality_only.tolist() == pytest.approx([0.5, 0.5])
    assert residual[1] > residual[0]
    assert residual[1] > quality_only[1]


def test_selector_breaks_equal_weight_ties_by_sample_index() -> None:
    candidates = (
        CoverageCandidate(
            prompt_id="p1",
            sample_index=5,
            text="later",
            quality_score=0.8,
            embedding=(1.0, 0.0),
        ),
        CoverageCandidate(
            prompt_id="p1",
            sample_index=2,
            text="earlier",
            quality_score=0.8,
            embedding=(1.0, 0.0),
        ),
    )
    probes = (StudentProbe(prompt_id="p1", sample_index=0, embedding=(0.0, 1.0)),)

    selected = select_residual_target(
        candidates,
        probes,
        parameters=CoverageResidualParameters(
            kappa=8.0, gamma=1.0, epsilon=1e-3, minimum_quality=0.6
        ),
    )

    assert selected.candidate.sample_index == 2
    assert selected.candidate.text == "earlier"


def test_artifact_records_selected_target_parameters_and_source_hashes() -> None:
    candidates = tuple(
        CoverageCandidate(
            prompt_id="p1",
            sample_index=index,
            text=f"teacher-{index}",
            quality_score=quality,
            embedding=embedding,
        )
        for index, (quality, embedding) in enumerate(((0.9, (1.0, 0.0)), (0.9, (0.0, 1.0))))
    )
    probes = (StudentProbe(prompt_id="p1", sample_index=0, embedding=(1.0, 0.0)),)
    parameters = CoverageResidualParameters(kappa=8.0, gamma=1.0, epsilon=1e-3, minimum_quality=0.6)

    artifact = build_coverage_residual_artifact(
        candidates,
        probes,
        parameters=parameters,
        source_hashes={"teacher": "a" * 64, "student": "b" * 64},
    )

    assert artifact["schema_version"] == 1
    assert artifact["targets"]["p1"]["best1"] == ["teacher-1"]
    assert artifact["targets"]["p1"]["all8"] == ["teacher-0", "teacher-1"]
    assert artifact["provenance"]["selector"] == parameters.model_dump(mode="json")
    assert artifact["provenance"]["source_hashes"] == {
        "teacher": "a" * 64,
        "student": "b" * 64,
    }
    assert artifact["audit"]["p1"]["selected_sample_index"] == 1


def test_selector_fails_when_no_teacher_candidate_passes_validity() -> None:
    candidates = (
        CoverageCandidate(
            prompt_id="p1",
            sample_index=0,
            text="invalid",
            quality_score=0.2,
            embedding=(1.0, 0.0),
        ),
    )
    probes = (StudentProbe(prompt_id="p1", sample_index=0, embedding=(0.0, 1.0)),)

    with pytest.raises(ValueError, match="valid teacher"):
        select_residual_target(
            candidates,
            probes,
            parameters=CoverageResidualParameters(
                kappa=8.0, gamma=1.0, epsilon=1e-3, minimum_quality=0.6
            ),
        )


def test_cli_writes_loadable_content_bound_target_artifact(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher.jsonl"
    student_path = tmp_path / "student.jsonl"
    output_path = tmp_path / "targets.json"
    teacher_records = [
        CoverageCandidate(
            prompt_id="p1",
            sample_index=0,
            text="covered",
            quality_score=0.9,
            embedding=(1.0, 0.0),
        ),
        CoverageCandidate(
            prompt_id="p1",
            sample_index=1,
            text="missing",
            quality_score=0.9,
            embedding=(0.0, 1.0),
        ),
    ]
    student_records = [
        StudentProbe(prompt_id="p1", sample_index=0, embedding=(1.0, 0.0)),
    ]
    teacher_path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in teacher_records),
        encoding="utf-8",
    )
    student_path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in student_records),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/derive_coverage_residual_targets.py",
            "--teacher-bank",
            str(teacher_path),
            "--student-bank",
            str(student_path),
            "--output",
            str(output_path),
            "--kappa",
            "8",
            "--gamma",
            "1",
            "--minimum-quality",
            "0.6",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["targets"]["p1"]["best1"] == ["missing"]
    assert payload["provenance"]["source_hashes"] == {
        "student_bank": __import__("hashlib").sha256(student_path.read_bytes()).hexdigest(),
        "teacher_bank": __import__("hashlib").sha256(teacher_path.read_bytes()).hexdigest(),
    }


def test_cli_rejects_undeclared_parameter_grid(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/derive_coverage_residual_targets.py",
            "--teacher-bank",
            str(tmp_path / "missing-teacher"),
            "--student-bank",
            str(tmp_path / "missing-student"),
            "--output",
            str(tmp_path / "output"),
            "--kappa",
            "9",
            "--gamma",
            "1",
            "--minimum-quality",
            "0.6",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
