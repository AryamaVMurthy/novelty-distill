import json
import subprocess
import sys

import pytest
import yaml

from novelty_distill.training.distillm import DistiLLMRunSpec
from novelty_distill.training.scale_config import (
    build_promoted_training_manifest,
    render_scale_training_config,
)
from novelty_distill.training.trl import TRLRunSpec


def test_trl_scale_config_preserves_method_controls_and_scales_exposures() -> None:
    base = yaml.safe_load(open("configs/training/gkd_tomato1k.yaml", encoding="utf-8"))

    rendered = render_scale_training_config(
        base, backend="trl", baseline_id="D2", train_size=5000, seed=29
    )

    spec = TRLRunSpec.model_validate(rendered)
    assert spec.max_examples == 5000
    assert spec.max_steps == 625
    assert spec.teacher_targets.name == "teacher-targets-tomato5000-v1.json"
    assert spec.temperature == base["temperature"]


def test_distillm_scale_config_keeps_ten_validation_stages() -> None:
    base = yaml.safe_load(open("configs/training/distillm_tomato1k.yaml", encoding="utf-8"))

    rendered = render_scale_training_config(
        base, backend="distillm", baseline_id="C3", train_size=20000, seed=43
    )

    spec = DistiLLMRunSpec.model_validate(rendered)
    assert spec.max_examples == 20000
    assert spec.dev_examples == 800
    assert spec.max_steps == 5000
    assert spec.validation_interval == 500
    assert spec.output_dir.name.endswith("l896-4gpu")


def test_small_replication_uses_separate_student_teacher_and_target_artifacts() -> None:
    base = yaml.safe_load(open("configs/training/gkd_tomato1k.yaml", encoding="utf-8"))

    rendered = render_scale_training_config(
        base,
        backend="trl",
        baseline_id="D1",
        train_size=1000,
        seed=17,
        model_profile="replication",
    )

    spec = TRLRunSpec.model_validate(rendered)
    assert spec.model == "Qwen/Qwen3-1.7B"
    assert spec.teacher_model == "Qwen/Qwen3-8B"
    assert spec.teacher_targets.name == "teacher-targets-qwen3-8b-tomato1000-v1.json"
    assert spec.output_dir.name == "D1-qwen1p7b-tomato1000-seed17"


def test_promoted_manifest_binds_every_method_seed_to_job_and_deployable_artifact() -> None:
    result = build_promoted_training_manifest(
        model_profile="main",
        train_size=20_000,
        promoted_indices=(0, 5, 12),
        seeds=(17, 29),
        training_dependencies={
            (0, 17): "500_0",
            (5, 17): "501",
            (12, 17): "502",
            (0, 29): "503_0",
            (5, 29): "504",
            (12, 29): "505",
        },
    )

    assert result["baseline_ids"] == ["B1", "B4", "C3"]
    assert len(result["runs"]) == 6
    by_identity = {(run["baseline_id"], run["seed"]): run for run in result["runs"]}
    assert by_identity[("B1", 17)]["lora_path"].endswith("checkpoints/B1-tomato20000-seed17/final")
    assert by_identity[("B4", 29)]["model_path"].endswith("checkpoints/B4-tomato20000-seed29")
    assert by_identity[("C3", 17)]["model_path"].endswith(
        "checkpoints/C3-tomato20000-seed17-l896-4gpu/5000"
    )
    assert by_identity[("C3", 17)]["model_dtype"] == "bfloat16"
    assert len({run["evaluation_id"] for run in result["runs"]}) == 6

    with pytest.raises(ValueError, match="dependency mapping"):
        build_promoted_training_manifest(
            model_profile="main",
            train_size=20_000,
            promoted_indices=(0, 5),
            seeds=(17,),
            training_dependencies={(0, 17): "500_0"},
        )


def test_promoted_manifest_cli_persists_exact_submission_graph(tmp_path) -> None:
    output = tmp_path / "submission.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/render_promoted_training_manifest.py",
            "--model-profile",
            "main",
            "--train-size",
            "5000",
            "--indices",
            "0,5",
            "--seeds",
            "17",
            "--dependency",
            "0:17=700_0",
            "--dependency",
            "5:17=701",
            "--output",
            str(output),
        ],
        check=True,
    )

    result = json.loads(output.read_text())
    assert result["repository_commit"]
    assert [run["training_dependency"] for run in result["runs"]] == ["700_0", "701"]
