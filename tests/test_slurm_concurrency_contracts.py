from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_training_environment_mutations_are_serialized() -> None:
    script = (ROOT / "slurm" / "train_smoke.sbatch").read_text(encoding="utf-8")

    assert 'exec 8>"${scratch_root}/locks/venv-${training_backend}.lock"' in script
    assert script.index("flock -x 8") < script.index("uv pip sync")
    assert script.index("uv pip install") < script.index("flock -u 8")


def test_shared_inference_environment_mutations_are_serialized() -> None:
    for name in (
        "sglang_smoke.sbatch",
        "score_teacher.sbatch",
        "evaluate_student.sbatch",
        "prepare_human_control.sbatch",
        "cluster_teacher.sbatch",
    ):
        script = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert 'exec 8>"${scratch_root}/locks/venv-inference.lock"' in script, name
        assert script.index("flock -x 8") < script.index("uv pip"), name
        assert script.rindex("uv pip") < script.index("flock -u 8"), name


def test_shared_data_environment_mutations_are_serialized() -> None:
    for name in ("validate_teacher_targets.sbatch", "analyze_evaluation_matrix.sbatch"):
        script = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert 'exec 8>"${scratch_root}/locks/venv-data.lock"' in script, name
        assert script.index("flock -x 8") < script.index("uv pip"), name
        assert script.rindex("uv pip") < script.index("flock -u 8"), name
