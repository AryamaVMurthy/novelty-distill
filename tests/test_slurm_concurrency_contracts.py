from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_cpu_only_jobs_stay_below_turing_gpu_billing_threshold() -> None:
    for path in sorted((ROOT / "slurm").glob("*.sbatch")):
        script = path.read_text(encoding="utf-8")
        if "#SBATCH --gres=gpu:" in script:
            continue
        cpu_lines = [
            line for line in script.splitlines() if line.startswith("#SBATCH --cpus-per-task=")
        ]
        if cpu_lines:
            assert int(cpu_lines[0].partition("=")[2]) <= 2, path.name


def test_training_environment_mutations_are_serialized() -> None:
    script = (ROOT / "slurm" / "train_smoke.sbatch").read_text(encoding="utf-8")

    assert 'exec 8>"${scratch_root}/locks/venv-${training_backend}.lock"' in script
    assert script.index("flock -x 8") < script.index("uv pip sync")
    assert script.index("uv pip install") < script.index("flock -u 8")


def test_training_retries_reject_concurrent_checkpoint_writers() -> None:
    script = (ROOT / "slurm" / "train_smoke.sbatch").read_text(encoding="utf-8")

    assert 'run_lock_key="$(printf \'%s\' "${metadata_path}" | sha256sum' in script
    assert 'exec 7>"${scratch_root}/locks/training-run-${run_lock_key}.lock"' in script
    assert script.index("if ! flock -n 7") < script.index("scripts/check_training_status.py")
    assert "flock -u 7" not in script


def test_scale_training_matrix_renders_frozen_exposure_configs() -> None:
    script = (ROOT / "slurm" / "train_smoke.sbatch").read_text(encoding="utf-8")

    assert '"${matrix_mode}" == "tomato_scale"' in script
    assert "scripts/render_scale_training_config.py" in script
    assert '--train-size "${scale_train_size}"' in script
    assert '--seed "${scale_seed}" \\\n    --model-profile "${model_profile}"' in script
    assert '--model-profile "${model_profile}"' in script


def test_training_can_select_a_validated_exploratory_registry() -> None:
    script = (ROOT / "slurm" / "train_smoke.sbatch").read_text(encoding="utf-8")

    assert 'baseline_registry="${BASELINE_REGISTRY:-configs/baselines.yaml}"' in script
    assert '--registry "${repo_dir}/${baseline_registry}"' in script


def test_shared_inference_environment_mutations_are_serialized() -> None:
    for name in (
        "sglang_smoke.sbatch",
        "score_teacher.sbatch",
        "evaluate_student.sbatch",
        "prepare_human_control.sbatch",
        "cluster_teacher.sbatch",
        "validate_generation_run.sbatch",
        "validate_score_run.sbatch",
        "merge_teacher_clusters.sbatch",
        "bootstrap_generation_run.sbatch",
    ):
        script = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert 'exec 8>"${scratch_root}/locks/venv-inference.lock"' in script, name
        assert script.index("flock -x 8") < script.index("uv pip"), name
        assert script.rindex("uv pip") < script.index("flock -u 8"), name


def test_official_evaluation_serializes_each_isolated_environment() -> None:
    script = (ROOT / "slurm" / "evaluate_official.sbatch").read_text(encoding="utf-8")

    assert 'exec 8>"${scratch_root}/locks/venv-inference.lock"' in script
    assert 'exec 7>"${scratch_root}/locks/venv-noveltybench.lock"' in script
    assert 'exec 7>"${scratch_root}/locks/venv-hypospace.lock"' in script
    assert script.count("flock -x 7") == 2
    assert script.count("flock -u 7") == 2


def test_shared_data_environment_mutations_are_serialized() -> None:
    for name in ("validate_teacher_targets.sbatch", "analyze_evaluation_matrix.sbatch"):
        script = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert 'exec 8>"${scratch_root}/locks/venv-data.lock"' in script, name
        assert script.index("flock -x 8") < script.index("uv pip"), name
        assert script.rindex("uv pip") < script.index("flock -u 8"), name


def test_official_combiner_serializes_inference_environment() -> None:
    script = (ROOT / "slurm" / "combine_official_results.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'exec 8>"${scratch_root}/locks/venv-inference.lock"' in script
    assert script.index("flock -x 8") < script.index("uv pip install")
    assert script.index("uv pip install") < script.index("flock -u 8")
