import yaml

from novelty_distill.training.distillm import DistiLLMRunSpec
from novelty_distill.training.scale_config import render_scale_training_config
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
