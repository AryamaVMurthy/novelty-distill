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


def test_parallel_training_consumers_import_from_source_during_editable_reinstalls() -> None:
    script = (ROOT / "slurm" / "train_smoke.sbatch").read_text(encoding="utf-8")

    export = 'export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"'
    assert export in script
    assert script.index(export) < script.index("flock -u 8")


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


def test_shared_inference_consumers_import_from_source_during_editable_reinstalls() -> None:
    """Parallel array tasks must survive another task replacing the editable wheel."""

    for name in (
        "sglang_smoke.sbatch",
        "score_teacher.sbatch",
        "evaluate_student.sbatch",
        "validate_generation_run.sbatch",
        "validate_score_run.sbatch",
    ):
        script = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        export = 'export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"'
        assert export in script, name
        assert script.index(export) < script.index("flock -u 8"), name


def test_official_evaluation_serializes_each_isolated_environment() -> None:
    script = (ROOT / "slurm" / "evaluate_official.sbatch").read_text(encoding="utf-8")

    # Each worker owns a job-specific environment; concurrent GPU workers do
    # not mutate one shared venv or need a global install lock.
    assert "inference-${SLURM_JOB_ID}" in script
    assert "noveltybench-${SLURM_JOB_ID}" in script
    assert "hypospace-${SLURM_JOB_ID}" in script


def test_shared_data_environment_mutations_are_serialized() -> None:
    for name in (
        "validate_teacher_targets.sbatch",
        "analyze_evaluation_matrix.sbatch",
        "prepare_deepinfra_calibration.sbatch",
    ):
        script = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert 'exec 8>"${scratch_root}/locks/venv-data.lock"' in script, name
        assert script.index("flock -x 8") < script.index("uv pip"), name
        assert script.rindex("uv pip") < script.index("flock -u 8"), name


def test_deepinfra_packet_preparation_imports_from_source() -> None:
    script = (ROOT / "slurm" / "prepare_deepinfra_calibration.sbatch").read_text(encoding="utf-8")
    export = 'export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"'
    assert export in script
    assert script.index(export) < script.index("flock -u 8")


def test_official_combiner_serializes_inference_environment() -> None:
    script = (ROOT / "slurm" / "combine_official_results.sbatch").read_text(encoding="utf-8")
    assert "inference-${SLURM_JOB_ID}" in script
    assert 'uv pip sync --python "${venv_dir}/bin/python"' in script


def test_final_adapter_stage_releases_only_after_byte_identity_gate() -> None:
    script = (ROOT / "slurm" / "stage_final_adapter_and_release.sbatch").read_text(encoding="utf-8")

    assert "adapter_config.json" in script
    assert "adapter_model.safetensors" in script
    assert script.count("hash_model_artifact.py") == 2
    assert 'if [[ "${destination_identity}" != "${source_identity}" ]]' in script
    assert script.rindex('scontrol release "${release_job_id}"') > script.rindex(
        'if [[ "${destination_identity}" != "${source_identity}" ]]'
    )
    assert "#SBATCH --gres=gpu:" not in script


def test_evaluation_repatriation_releases_only_after_every_declared_hash_gate() -> None:
    script = (ROOT / "slurm" / "repatriate_compact_evaluations_and_release.sbatch").read_text(
        encoding="utf-8"
    )

    loop_start = script.index('for spec in "${specs[@]}"')
    loop_end = script.index("done", loop_start)
    release = script.index('scontrol release "${release_job_id}"')
    assert loop_start < loop_end < release
    assert 'if [[ "${destination_hash}" != "${source_hash}" ]]' in script
    assert "temporal-k4-corrected-v2.json" in script
    assert "#SBATCH --gres=gpu:" not in script
