from pathlib import Path

import yaml

from novelty_distill.config import load_baseline_registry
from novelty_distill.training.gem import GEMRunSpec
from novelty_distill.training.trl import TRLRunSpec


def test_exposure_sensitivity_registry_is_separate_and_matched() -> None:
    registry = load_baseline_registry(Path("configs/exposure_sensitivity_baselines.yaml"))
    by_id = {baseline.id: baseline for baseline in registry.baselines}

    assert tuple(by_id) == (
        "B2a-4x",
        "B2b-4x",
        "B2c-4x",
        "B3-4x",
        "B4-4x",
        "F2-random4",
    )
    assert [by_id[name].target_view for name in by_id] == [
        "random1",
        "best1",
        "mode1",
        "diverse4",
        "diverse4",
        "random4",
    ]
    assert all(baseline.family == "sft" for baseline in by_id.values())


def test_exposure_sensitivity_configs_use_exactly_four_thousand_exposures() -> None:
    sft = TRLRunSpec.model_validate(
        yaml.safe_load(Path("configs/training/sft_tomato1k_4x.yaml").read_text())
    )
    random4 = TRLRunSpec.model_validate(
        yaml.safe_load(Path("configs/training/sft_tomato1k_random4.yaml").read_text())
    )
    gem = GEMRunSpec.model_validate(
        yaml.safe_load(Path("configs/training/gem_tomato1k_4x.yaml").read_text())
    )

    for spec in (sft, random4, gem):
        assert spec.max_examples == 1000
        exposures = (
            spec.max_steps
            * spec.per_device_train_batch_size
            * spec.gradient_accumulation_steps
        )
        assert exposures == 4000
    assert sft.output_dir.name.endswith("exposure4x-seed17")
    assert random4.baseline_id == "F2-random4"
    assert random4.output_dir.name.endswith("exposure4x-seed17")
    assert gem.output_dir.name.endswith("exposure4x-seed17")


def test_exposure_sensitivity_launcher_is_low_priority_bounded_and_post_primary() -> None:
    launcher = Path("scripts/submit_exposure_sensitivity.sh").read_text(encoding="utf-8")
    training = Path("slurm/train_smoke.sbatch").read_text(encoding="utf-8")

    assert 'start_after_job_id="${START_AFTER_JOB_ID:?' in launcher
    assert "--array=0-4%2" in launcher
    assert "--nice=10000" in launcher
    assert 'training_passes="${TRAINING_PASSES:-2}"' in launcher
    assert "BASELINE_MATRIX=exposure1k" in launcher
    assert 'matrix_mode}" == "exposure1k"' in training
    assert "configs/exposure_sensitivity_baselines.yaml" in training


def test_exposure_evaluation_launcher_binds_every_checkpoint_and_array_task() -> None:
    launcher = Path("scripts/submit_exposure_sensitivity_evaluations.sh").read_text(
        encoding="utf-8"
    )

    assert 'training_job_id="${TRAINING_JOB_ID:?' in launcher
    assert "methods=(B2a-4x B2b-4x B2c-4x B3-4x B4-4x)" in launcher
    assert 'TRAINING_DEPENDENCY="${training_job_id}_${index}"' in launcher
    assert launcher.count("scripts/submit_model_evaluation.sh") == 1
    assert 'RUN_TASTE="1"' in launcher
    assert 'LORA_PATH="${lora_path}"' in launcher
    assert 'generation_config="configs/generation/eval_qwen3_4b.yaml"' in launcher


def test_random4_launcher_is_low_priority_resumable_and_fully_evaluated() -> None:
    launcher = Path("scripts/submit_random4_sensitivity.sh").read_text(encoding="utf-8")

    assert 'start_after_job_id="${START_AFTER_JOB_ID:?' in launcher
    assert 'training_passes="${TRAINING_PASSES:-2}"' in launcher
    assert "F2-random4" in launcher
    assert "configs/training/sft_tomato1k_random4.yaml" in launcher
    assert "configs/exposure_sensitivity_baselines.yaml" in launcher
    assert "--nice=10000" in launcher
    assert "scripts/submit_model_evaluation.sh" in launcher
    assert 'RUN_TASTE="1"' in launcher
    assert 'TRAINING_DEPENDENCY="${training_job}"' in launcher


def test_exposure_analysis_is_separate_audited_and_uses_declared_contrasts() -> None:
    analysis = Path("slurm/analyze_exposure_sensitivity.sbatch").read_text(encoding="utf-8")
    config = yaml.safe_load(
        Path("configs/evaluation/exposure_sensitivity_contrasts.yaml").read_text()
    )

    assert "configs/exposure_sensitivity_baselines.yaml" in analysis
    assert "scripts/audit_training_matrix.py" in analysis
    assert "scripts/analyze_contrast_matrix.py" in analysis
    assert "scripts/analyze_research_taste.py" in analysis
    assert "scripts/export_wandb_snapshot.py" in analysis
    assert config["status"] == "secondary_sensitivity"
    contrast_ids = {contrast["id"] for contrast in config["contrasts"]}
    assert {
        "B3-4x-vs-B2a-4x",
        "B3-4x-vs-B2b-4x",
        "B3-4x-vs-B2c-4x",
        "B4-4x-vs-B3-4x",
        "B3-4x-vs-B3",
        "F2-random4-vs-B2a-4x",
        "B3-4x-vs-F2-random4",
    } <= contrast_ids


def test_exposure_analysis_controller_fails_closed_and_persists_graph() -> None:
    controller = Path("slurm/submit_exposure_sensitivity_analysis.sbatch").read_text(
        encoding="utf-8"
    )

    assert 'primary_controller_job_id="${PRIMARY_CONTROLLER_JOB_ID:?' in controller
    assert 'exposure_evaluation_job_ids="${EXPOSURE_EVALUATION_JOB_IDS:?' in controller
    assert 'exposure_taste_job_ids="${EXPOSURE_TASTE_JOB_IDS:?' in controller
    assert '"analysis"' in controller
    assert "reversed(lines)" in controller
    assert 'expected_exposure_jobs="${EXPECTED_EXPOSURE_JOBS:-6}"' in controller
    assert 'dependency="afterok:${all_dependencies}"' in controller
    assert "slurm/analyze_exposure_sensitivity.sbatch" in controller
    assert "exposure-sensitivity-analysis.json" in controller
    assert "os.replace" in controller
