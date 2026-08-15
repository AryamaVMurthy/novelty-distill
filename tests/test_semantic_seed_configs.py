from pathlib import Path

from novelty_distill.config import load_baseline_registry
from novelty_distill.generation.sglang import GenerationSpec
from novelty_distill.training.trl import load_trl_run_spec

TRAINING_CONFIGS = {
    "SS0-C1": "semantic_seed_ss0_c1.yaml",
    "SS0D-DIVERSE": "semantic_seed_ss0d_diverse.yaml",
    "SS1-CR": "semantic_seed_ss1_cr.yaml",
    "SS2-GSC": "semantic_seed_ss2_gsc.yaml",
    "SS3-TL": "semantic_seed_ss3_tl.yaml",
    "SS4-GSC-TL": "semantic_seed_ss4_gsc_tl.yaml",
}


def test_semantic_seed_registry_freezes_six_forward_kl_candidates() -> None:
    registry = load_baseline_registry(Path("configs/semantic_seed_baselines.yaml"))

    assert [baseline.id for baseline in registry.baselines] == list(TRAINING_CONFIGS)
    assert all(baseline.backend == "trl_gkd" for baseline in registry.baselines)
    assert all(baseline.family == "off_policy" for baseline in registry.baselines)
    assert all(baseline.trajectory_source == "teacher" for baseline in registry.baselines)
    assert all(baseline.divergence == "forward_kl" for baseline in registry.baselines)
    assert all(baseline.lmbda == 0 and baseline.beta == 0 for baseline in registry.baselines)
    views = {baseline.id: baseline.target_view for baseline in registry.baselines}
    assert views == {
        "SS0-C1": "best1",
        "SS0D-DIVERSE": "diverse4",
        "SS1-CR": "best1",
        "SS2-GSC": "diverse4",
        "SS3-TL": "best1",
        "SS4-GSC-TL": "diverse4",
    }


def test_short_training_configs_match_models_exposure_and_data() -> None:
    specs = {
        baseline_id: load_trl_run_spec(Path("configs/training") / filename)
        for baseline_id, filename in TRAINING_CONFIGS.items()
    }

    assert {spec.baseline_id for spec in specs.values()} == set(TRAINING_CONFIGS)
    assert {spec.model for spec in specs.values()} == {"Qwen/Qwen3-4B"}
    assert {spec.revision for spec in specs.values()} == {
        "1cfa9a7208912126459214e8b04321603b3df60c"
    }
    assert {spec.teacher_model for spec in specs.values()} == {"Qwen/Qwen3-14B"}
    assert {spec.teacher_revision for spec in specs.values()} == {
        "40c069824f4251a91eefaf281ebe4c544efd3e18"
    }
    assert {spec.input for spec in specs.values()} == {
        Path("data/semantic-seed-train-128.jsonl")
    }
    assert {spec.max_examples for spec in specs.values()} == {128}
    assert {
        spec.max_steps * spec.per_device_train_batch_size * spec.gradient_accumulation_steps
        for spec in specs.values()
    } == {512}
    assert all(spec.max_steps == 64 for spec in specs.values())


def test_only_declared_candidates_enable_seed_and_lookahead() -> None:
    specs = {
        baseline_id: load_trl_run_spec(Path("configs/training") / filename)
        for baseline_id, filename in TRAINING_CONFIGS.items()
    }

    assert {name for name, spec in specs.items() if spec.input_seed is not None} == {
        "SS2-GSC",
        "SS4-GSC-TL",
    }
    assert {name for name, spec in specs.items() if spec.teacherless_weight > 0} == {
        "SS3-TL",
        "SS4-GSC-TL",
    }
    for name in ("SS2-GSC", "SS4-GSC-TL"):
        assert specs[name].input_seed.model_dump(mode="json") == {
            "dimensions": 8,
            "bins": 5,
            "scale": 1.0,
            "salt": "semantic-seed-v1",
        }
    for name in ("SS3-TL", "SS4-GSC-TL"):
        assert specs[name].teacherless_weight == 0.1
        assert specs[name].teacherless_neutral_token == "!"
    assert specs["SS1-CR"].teacher_targets == Path(
        "data/teacher-targets-semantic-residual-k16-g1.json"
    )


def test_short_generation_configs_isolate_ordinary_and_seeded_decoding() -> None:
    import yaml

    ordinary = GenerationSpec.model_validate(
        yaml.safe_load(Path("configs/generation/semantic_seed_short.yaml").read_text())
    )
    seeded = GenerationSpec.model_validate(
        yaml.safe_load(Path("configs/generation/semantic_seed_short_seeded.yaml").read_text())
    )

    assert ordinary.model == seeded.model == "Qwen/Qwen3-4B"
    assert ordinary.temperature == seeded.temperature == 0.2
    assert ordinary.samples_per_prompt == seeded.samples_per_prompt == 8
    assert ordinary.input_seed is None
    assert seeded.input_seed is not None
    assert seeded.input_seed.dimensions == 8


def test_combined_gpu_smoke_exercises_both_new_training_paths() -> None:
    spec = load_trl_run_spec(
        Path("configs/training/semantic_seed_ss4_gsc_tl_smoke.yaml")
    )

    assert spec.baseline_id == "SS4-GSC-TL"
    assert spec.max_examples == spec.max_steps == 1
    assert spec.per_device_train_batch_size == spec.gradient_accumulation_steps == 1
    assert spec.input_seed is not None
    assert spec.teacherless_weight == 0.1
    assert spec.teacherless_neutral_token == "!"
