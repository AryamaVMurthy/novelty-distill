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


def test_sglang_job_forwards_deterministic_prompt_shard_coordinates() -> None:
    script = Path("slurm/sglang_smoke.sbatch").read_text(encoding="utf-8")

    assert 'generation_num_shards="${GENERATION_NUM_SHARDS:-1}"' in script
    assert 'generation_shard_index="${GENERATION_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"' in script
    assert script.count('--num-shards "${generation_num_shards}"') == 3
    assert script.count('--shard-index "${generation_shard_index}"') == 3
    assert 'input_name="${INPUT_NAME:-}"' in script


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
    assert 'eval_id="${EVAL_ID:-base-qwen3-4b}"' in script
    assert '--lora-paths "${lora_name}=${lora_path}"' in script
    assert 'base_served_model="novelty-base"' in script
    assert 'result_root="${scratch_root}/evaluations/official/${eval_id}"' in script


def test_research_taste_job_is_resumable_and_uses_the_pinned_annotator() -> None:
    script = Path("slurm/annotate_research_taste.sbatch").read_text(encoding="utf-8")

    assert "scripts/check_research_taste_status.py" in script
    assert "scripts/annotate_research_taste.py" in script
    assert "configs/evaluation/research_taste.yaml" in script
    assert 'annotator_model="Qwen/Qwen3-32B-FP8"' in script
    assert 'flock -x 8' in script


def test_quality_score_job_forwards_deterministic_prompt_shard_coordinates() -> None:
    script = Path("slurm/score_teacher.sbatch").read_text(encoding="utf-8")

    assert 'score_num_shards="${SCORE_NUM_SHARDS:-1}"' in script
    assert 'score_shard_index="${SCORE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"' in script
    assert script.count('--num-shards "${score_num_shards}"') == 2
    assert script.count('--shard-index "${score_shard_index}"') == 2
    assert 'prompts_name="${PROMPTS_NAME:-}"' in script


def test_global_generation_gate_ignores_worker_sharding() -> None:
    script = Path("slurm/validate_generation_run.sbatch").read_text(encoding="utf-8")

    assert "scripts/check_generation_status.py" in script
    assert "--num-shards 1" in script
    assert "EXPECTED_PROMPTS" in script


def test_global_score_gate_ignores_worker_sharding() -> None:
    script = Path("slurm/validate_score_run.sbatch").read_text(encoding="utf-8")

    assert "scripts/check_score_status.py" in script
    assert "--num-shards 1" in script
    assert "EXPECTED_PROMPTS" in script


def test_teacher_clustering_supports_gpu_partitions_and_strict_merge() -> None:
    cluster = Path("slurm/cluster_teacher.sbatch").read_text(encoding="utf-8")
    merge = Path("slurm/merge_teacher_clusters.sbatch").read_text(encoding="utf-8")

    assert 'cluster_num_shards="${CLUSTER_NUM_SHARDS:-1}"' in cluster
    assert '--num-shards "${cluster_num_shards}"' in cluster
    assert "scripts/merge_teacher_clusters.py" in merge
    assert '--expected-prompts "${expected_prompts}"' in merge


def test_scale_teacher_launcher_uses_four_gpu_arrays_and_global_gates() -> None:
    script = Path("scripts/submit_teacher_scale.sh").read_text(encoding="utf-8")

    assert 'num_gpu_shards="${NUM_GPU_SHARDS:-4}"' in script
    assert script.count('--array="0-$((num_gpu_shards - 1))%${num_gpu_shards}"') == 3
    assert "slurm/validate_generation_run.sbatch" in script
    assert "slurm/bootstrap_generation_run.sbatch" in script
    assert "slurm/validate_score_run.sbatch" in script
    assert "slurm/merge_teacher_clusters.sbatch" in script
    assert "slurm/validate_teacher_targets.sbatch" in script
    assert 'teacher_profile="${TEACHER_PROFILE:-main}"' in script
    assert "configs/generation/teacher_qwen3_8b.yaml" in script
    assert 'generation_id="teacher8b-${train_size}-v1"' in script


def test_promoted_training_launcher_separates_special_resource_backends() -> None:
    script = Path("scripts/submit_promoted_training.sh").read_text(encoding="utf-8")

    assert 'promoted_indices="${PROMOTED_INDICES:?' in script
    assert '--array="${adapter_spec}%4"' in script
    assert "--array=5" in script
    assert "--array=12" in script
    assert "--gres=gpu:4" in script
    assert "BASELINE_MATRIX=tomato_scale" in script
    assert 'model_profile="${MODEL_PROFILE:-main}"' in script


def test_official_model_launcher_runs_full_suite_on_four_gpu_jobs() -> None:
    script = Path("scripts/submit_official_model_evaluation.sh").read_text(encoding="utf-8")

    assert "EVAL_SUITE=noveltybench,EVAL_LIMIT=100" in script
    assert "domains=(causal 3d boolean)" in script
    assert "limits=(61 9 35)" in script
    assert "NUM_GENERATIONS=10" in script
    assert "slurm/combine_official_results.sbatch" in script


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
