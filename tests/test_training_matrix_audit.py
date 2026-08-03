import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from novelty_distill.training.audit import audit_training_matrix


def _write_registry(path: Path, *, include_second_runnable: bool = False) -> None:
    second = (
        """
  - id: B2a
    name: teacher-sft
    family: sft
    backend: trl_sft
    official_source: example/sft
    trajectory_source: teacher
    target_view: random1
"""
        if include_second_runnable
        else ""
    )
    path.write_text(
        f"""\
schema_version: 1
baselines:
  - id: A0
    name: control
    family: control
    backend: evaluation
    official_source: example/control
  - id: B1
    name: human-sft
    family: sft
    backend: trl_sft
    official_source: example/sft
    trajectory_source: human
    target_view: human
  - id: E1
    name: unavailable-control
    family: self_distillation
    backend: opsd
    official_source: example/opsd
    execution_status: fail_closed
    non_executable_reason: official backend has no required mode
    trajectory_source: static
    target_view: human
    divergence: forward_kl
    teacher_context: privileged
    lmbda: 0.0
    beta: 0.0
{second}""",
        encoding="utf-8",
    )


def _write_complete_run(root: Path, registry: Path, training_input: Path) -> Path:
    final = root / "final"
    final.mkdir(parents=True)
    (final / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (final / "adapter_model.safetensors").write_bytes(b"real-weights")
    metadata = root / "run_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "baseline_id": "B1",
                "backend": "trl_sft",
                "model": "Qwen/Qwen3-4B",
                "revision": "a" * 40,
                "trajectory_source": "human",
                "target_view": "human",
                "divergence": "none",
                "dataset_revision": "b" * 40,
                "example_ids": ["one", "two"],
                "training_artifacts": {
                    "input": {
                        "path": str(training_input),
                        "size_bytes": training_input.stat().st_size,
                        "sha256": hashlib.sha256(training_input.read_bytes()).hexdigest(),
                    },
                    "registry": {
                        "path": str(registry),
                        "size_bytes": registry.stat().st_size,
                        "sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
                    },
                },
                "training_rows": 2,
                "optimizer_example_exposures": 2,
                "max_steps": 1,
                "seed": 17,
                "slurm_job_id": "42",
                "git_commit": "d" * 40,
                "metrics": {"train_loss": 1.25, "train_runtime": 4.0},
                "final_dir": str(final),
            }
        ),
        encoding="utf-8",
    )
    return metadata


def test_audit_proves_complete_runnable_matrix_and_records_fail_closed_control(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "baselines.yaml"
    _write_registry(registry)
    training_input = tmp_path / "train.jsonl"
    training_input.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")
    metadata = _write_complete_run(tmp_path / "B1", registry, training_input)

    result = audit_training_matrix(registry, {"B1": metadata})

    assert result["status"] == "complete"
    assert result["executable_baseline_ids"] == ["B1"]
    assert result["fail_closed_baselines"] == [
        {
            "id": "E1",
            "reason": "official backend has no required mode",
        }
    ]
    assert result["common_contract"] == {
        "dataset_revision": "b" * 40,
        "example_count": 2,
        "example_ids_sha256": result["common_contract"]["example_ids_sha256"],
        "input_sha256": hashlib.sha256(training_input.read_bytes()).hexdigest(),
        "input_size_bytes": training_input.stat().st_size,
        "model": "Qwen/Qwen3-4B",
        "revision": "a" * 40,
        "seed": 17,
        "optimizer_example_exposures": 2,
    }
    assert result["runs"]["B1"]["artifact_identity"].startswith("sha256:")
    assert result["runs"]["B1"]["training_rows"] == 2


def test_audit_cli_writes_the_validated_matrix_atomically(tmp_path: Path) -> None:
    registry = tmp_path / "baselines.yaml"
    _write_registry(registry)
    training_input = tmp_path / "train.jsonl"
    training_input.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")
    metadata = _write_complete_run(tmp_path / "B1", registry, training_input)
    output = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_training_matrix.py",
            "--registry",
            str(registry),
            "--input",
            f"B1={metadata}",
            "--output",
            str(output),
        ],
        check=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["executable_baseline_count"] == 1
    assert len(result["repository_commit"]) == 40


def test_audit_rejects_metadata_without_completed_training_evidence(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "baselines.yaml"
    _write_registry(registry)
    training_input = tmp_path / "train.jsonl"
    training_input.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")
    metadata = _write_complete_run(tmp_path / "B1", registry, training_input)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["metrics"] = {}
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="completed training evidence"):
        audit_training_matrix(registry, {"B1": metadata})


def test_final_analysis_requires_the_complete_training_matrix_audit() -> None:
    script = Path("slurm/analyze_evaluation_matrix.sbatch").read_text(encoding="utf-8")

    assert "scripts/audit_training_matrix.py" in script
    assert "training-matrix-audit.json" in script
    assert '--registry "${repo_dir}/configs/baselines.yaml"' in script
    assert '--input "${baseline_id}=${metadata_path}"' in script
    assert 'test -s "${output_dir}/training-matrix-audit.json"' in script


def test_promoted_audit_can_require_an_explicit_runnable_subset(tmp_path: Path) -> None:
    registry = tmp_path / "baselines.yaml"
    _write_registry(registry, include_second_runnable=True)
    training_input = tmp_path / "train.jsonl"
    training_input.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")
    metadata = _write_complete_run(tmp_path / "B1", registry, training_input)

    result = audit_training_matrix(
        registry,
        {"B1": metadata},
        required_baseline_ids=("B1",),
    )

    assert result["executable_baseline_ids"] == ["B1"]
