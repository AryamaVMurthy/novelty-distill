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
