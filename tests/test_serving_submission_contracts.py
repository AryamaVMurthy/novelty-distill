from pathlib import Path


def test_sglang_job_forwards_validated_model_dtype() -> None:
    script = Path("slurm/sglang_smoke.sbatch").read_text(encoding="utf-8")

    assert 'model_dtype="${MODEL_DTYPE:-auto}"' in script
    assert '"${model_dtype}" != auto' in script
    assert '--dtype "${model_dtype}"' in script


def test_sglang_job_binds_local_checkpoint_or_adapter_bytes() -> None:
    script = Path("slurm/sglang_smoke.sbatch").read_text(encoding="utf-8")

    assert "scripts/hash_model_artifact.py" in script
    assert (
        'artifact_identity_args=(--served-artifact-identity "${served_artifact_identity}")'
        in script
    )
    assert script.count('"${artifact_identity_args[@]}"') == 3


def test_distillm_matrix_evaluation_forces_bfloat16_serving() -> None:
    script = Path("slurm/submit_evaluation_matrix.sbatch").read_text(
        encoding="utf-8"
    )

    assert 'local model_dtype="${5:-auto}"' in script
    assert "MODEL_DTYPE=${model_dtype}" in script
    assert (
        'submit_chain C3 "${distillm_path}" configs/generation/eval_qwen3_4b.yaml "" bfloat16'
        in script
    )


def test_official_evaluation_supports_scratch_cached_dtype_override() -> None:
    script = Path("slurm/evaluate_official.sbatch").read_text(encoding="utf-8")

    assert 'model_dtype="${MODEL_DTYPE:-auto}"' in script
    assert 'TVM_FFI_CACHE_DIR="${scratch_root}/cache/tvm-ffi"' in script
    assert '--dtype "${model_dtype}"' in script


def test_research_taste_job_is_resumable_and_uses_the_pinned_annotator() -> None:
    script = Path("slurm/annotate_research_taste.sbatch").read_text(encoding="utf-8")

    assert "scripts/check_research_taste_status.py" in script
    assert "scripts/annotate_research_taste.py" in script
    assert "configs/evaluation/research_taste.yaml" in script
    assert 'annotator_model="Qwen/Qwen3-32B-FP8"' in script
    assert 'flock -x 8' in script


def test_final_controller_submits_research_taste_for_every_generation_family() -> None:
    script = Path("slurm/submit_evaluation_matrix.sbatch").read_text(encoding="utf-8")

    assert "taste_jobs=()" in script
    assert "submit_taste_chain" in script
    assert 'submit_taste_chain A1-temporal-k16-seed17000' in script
    assert 'submit_taste_chain A0-temporal-k16-seed17000' in script
    assert 'submit_taste_chain A3-temporal-k1' in script
    assert 'taste_jobs+=("${taste_job}")' in script
    assert 'taste_dependency="$(IFS=:; printf \'%s\' "${taste_jobs[*]}")"' in script


def test_final_analysis_keeps_taste_results_separate_from_primary_contrasts() -> None:
    script = Path("slurm/analyze_evaluation_matrix.sbatch").read_text(encoding="utf-8")

    assert "scripts/analyze_research_taste.py" in script
    assert "research-taste-secondary.json" in script
    assert "research-taste-findings.md" in script
    assert "scripts/prepare_research_taste_calibration.py" in script
    assert "research-taste-human-calibration.jsonl" in script
    assert "research-taste-human-calibration-key.json" in script
    assert "primary-contrasts.json" in script
