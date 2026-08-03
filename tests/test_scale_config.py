import json
import subprocess
import sys

import pytest
import yaml

from novelty_distill.training.distillm import DistiLLMRunSpec
from novelty_distill.training.gem import GEMRunSpec
from novelty_distill.training.opsd import OPSDRunSpec
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


def test_replication_manifest_uses_matching_served_model_and_generation_configs() -> None:
    result = build_promoted_training_manifest(
        model_profile="replication",
        train_size=1000,
        promoted_indices=(0, 5),
        seeds=(17,),
        training_dependencies={(0, 17): "800_0", (5, 17): "801"},
    )

    by_id = {run["baseline_id"]: run for run in result["runs"]}
    assert by_id["B1"]["served_model_name"] == "Qwen/Qwen3-1.7B"
    assert by_id["B1"]["generation_config"].endswith("eval_qwen3_1p7b_lora.yaml")
    assert by_id["B4"]["served_model_name"] == "Qwen/Qwen3-1.7B"
    assert by_id["B4"]["generation_config"].endswith("eval_qwen3_1p7b.yaml")


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


def test_every_scale_backend_profile_size_and_seed_has_valid_unique_contract() -> None:
    cases = (
        ("B1", "trl", "sft", TRLRunSpec),
        ("B2a", "trl", "sft", TRLRunSpec),
        ("B2b", "trl", "sft", TRLRunSpec),
        ("B2c", "trl", "sft", TRLRunSpec),
        ("B3", "trl", "sft", TRLRunSpec),
        ("B4", "gem", "gem", GEMRunSpec),
        ("C1-human", "trl", "gkd", TRLRunSpec),
        ("C1-best1", "trl", "gkd", TRLRunSpec),
        ("C1-diverse4", "trl", "gkd", TRLRunSpec),
        ("C2-human", "trl", "gkd", TRLRunSpec),
        ("C2-best1", "trl", "gkd", TRLRunSpec),
        ("C2-diverse4", "trl", "gkd", TRLRunSpec),
        ("C3", "distillm", "distillm", DistiLLMRunSpec),
        ("D1", "trl", "gkd", TRLRunSpec),
        ("D2", "trl", "gkd", TRLRunSpec),
        ("D3", "trl", "gkd", TRLRunSpec),
        ("E2", "opsd", "opsd", OPSDRunSpec),
        ("E3", "opsd", "opsd", OPSDRunSpec),
        ("E4", "opsd", "opsd", OPSDRunSpec),
    )
    output_dirs: set[str] = set()
    validated = 0
    for profile, sizes in (("main", (5000, 20000)), ("replication", (1000, 5000, 20000))):
        expected_model = "Qwen/Qwen3-4B" if profile == "main" else "Qwen/Qwen3-1.7B"
        for train_size in sizes:
            for seed in (17, 29, 43):
                for baseline_id, backend, stem, validator in cases:
                    base = yaml.safe_load(
                        open(f"configs/training/{stem}_tomato1k.yaml", encoding="utf-8")
                    )
                    rendered = render_scale_training_config(
                        base,
                        backend=backend,
                        baseline_id=baseline_id,
                        train_size=train_size,
                        seed=seed,
                        model_profile=profile,
                    )
                    spec = validator.model_validate(rendered)
                    output_dir = str(spec.output_dir)
                    assert output_dir not in output_dirs
                    output_dirs.add(output_dir)
                    model = rendered.get("student_model", rendered.get("model"))
                    assert model == expected_model
                    assert rendered["max_examples"] == train_size
                    exposures = (
                        rendered["max_steps"]
                        * rendered.get("batch_size", rendered.get("per_device_train_batch_size", 1))
                        * rendered.get("num_gpus", 1)
                        * rendered.get("gradient_accumulation_steps", 1)
                    )
                    assert train_size <= exposures < train_size + 8
                    validated += 1
    assert validated == 285
