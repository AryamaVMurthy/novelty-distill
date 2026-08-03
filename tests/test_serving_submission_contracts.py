import json
import subprocess
import sys
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
    assert 'model_path="${scratch_root}/${model_path}"' in script
    assert 'lora_path="${scratch_root}/${lora_path}"' in script


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
    assert 'model_path="${scratch_root}/${model_path}"' in script
    assert 'lora_path="${scratch_root}/${lora_path}"' in script


def test_research_taste_job_is_resumable_and_uses_the_pinned_annotator() -> None:
    script = Path("slurm/annotate_research_taste.sbatch").read_text(encoding="utf-8")

    assert "scripts/check_research_taste_status.py" in script
    assert "scripts/annotate_research_taste.py" in script
    assert "configs/evaluation/research_taste.yaml" in script
    assert 'annotator_model="Qwen/Qwen3-32B-FP8"' in script
    assert 'flock -x 8' in script
    assert 'taste_attempts="${TASTE_ATTEMPTS:-3}"' in script
    assert '--attempts "${taste_attempts}"' in script
    assert 'prompts_name="${PROMPTS_NAME:-}"' in script
    assert 'prompts_path="${scratch_root}/data/${prompts_name}"' in script
    annotator = Path("scripts/annotate_research_taste.py").read_text(encoding="utf-8")
    assert "def _request_annotation(" in annotator
    assert "for attempt in range(1, attempts + 1):" in annotator


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
    assert "scripts/render_promoted_training_manifest.py" in script
    assert 'submission_manifest="${SUBMISSION_MANIFEST:-}"' in script
    assert "${adapter_job}_${index}" in script


def test_official_model_launcher_runs_full_suite_on_four_gpu_jobs() -> None:
    script = Path("scripts/submit_official_model_evaluation.sh").read_text(encoding="utf-8")

    assert "EVAL_SUITE=noveltybench,EVAL_LIMIT=100" in script
    assert "domains=(causal 3d boolean)" in script
    assert "limits=(61 9 35)" in script
    assert "NUM_GENERATIONS=10" in script
    assert "slurm/combine_official_results.sbatch" in script


def test_single_model_temporal_launcher_can_chain_resumable_taste_annotation() -> None:
    script = Path("scripts/submit_model_evaluation.sh").read_text(encoding="utf-8")

    assert 'run_taste="${RUN_TASTE:-0}"' in script
    assert 'taste_passes="${TASTE_PASSES:-2}"' in script
    assert "slurm/annotate_research_taste.sbatch" in script
    assert '"taste_final":"%s"' in script
    assert '^[0-9]+(_[0-9]+)?$' in script


def test_compact_launcher_uses_four_shards_and_budget_matched_evaluation() -> None:
    launcher = Path("scripts/submit_compact_baselines.sh").read_text(encoding="utf-8")
    evaluation = Path("slurm/evaluate_student.sbatch").read_text(encoding="utf-8")

    assert 'num_gpu_shards="${NUM_GPU_SHARDS:-4}"' in launcher
    assert launcher.count('--array="0-3%4"') == 3
    assert "baseline_ids=(B1 B2b C1-best1 C2-best1 D1)" in launcher
    assert "TEACHER_SOURCE_SAMPLES_PER_PROMPT=16" in launcher
    assert "STUDENT_SOURCE_SAMPLES_PER_PROMPT=4" in launcher
    assert "A0-temporal-k4-from-k16" in launcher
    assert 'teacher_source_samples_per_prompt="${TEACHER_SOURCE_SAMPLES_PER_PROMPT:-' in evaluation
    assert '--teacher-source-samples-per-prompt' in evaluation


def test_promoted_evaluation_dry_run_binds_models_controls_taste_and_analysis(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "training.json"
    output = tmp_path / "evaluation.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_profile": "main",
                "train_size": 5000,
                "seeds": [17],
                "promoted_indices": [0],
                "baseline_ids": ["B1"],
                "target_name": "teacher-targets-tomato5000-v1.json",
                "target_gate_job_id": 900,
                "runs": [
                    {
                        "baseline_id": "B1",
                        "matrix_index": 0,
                        "seed": 17,
                        "backend": "trl",
                        "training_dependency": "901_0",
                        "metadata_path": "checkpoints/B1-tomato5000-seed17/run_metadata.json",
                        "model_path": "Qwen/Qwen3-4B",
                        "served_model_name": "Qwen/Qwen3-4B",
                        "model_revision": "revision",
                        "lora_path": "checkpoints/B1-tomato5000-seed17/final",
                        "model_dtype": "auto",
                        "generation_config": "configs/generation/eval_qwen3_4b_lora.yaml",
                        "evaluation_id": "B1-tomato5000-seed17-temporal-k16",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/submit_promoted_evaluations.py",
            "--manifest",
            str(manifest),
            "--teacher-score-job-id",
            "902",
            "--output",
            str(output),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["dry_run"] is True
    assert result["controls"]["A0"]["evaluation_id"] == "A0-tomato5000-temporal-k16"
    run = result["runs"][0]
    assert run["environment"]["RUN_TASTE"] == "1"
    assert run["environment"]["SERVED_MODEL_NAME"] == "Qwen/Qwen3-4B"
    assert run["environment"]["TRAINING_DEPENDENCY"] == "901_0"
    assert run["environment"]["LORA_PATH"].endswith("/final")
    assert result["analysis"]["environment"]["PROMOTED_METHODS"] == "B1"
    assert result["analysis"]["environment"]["TRAIN_SEEDS"] == "17"
    assert result["analysis"]["environment"]["CONTROL_A0_SCORE_ID"] == (
        "A0-temporal-k16-seed17000"
    )
    assert result["analysis"]["environment"]["CONTROL_A1_SCORE_ID"] == (
        "A1-temporal-k16-seed17000"
    )
    assert result["analysis"]["environment"]["CONTROL_A3_SCORE_ID"] == "A3-temporal-k1"
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_promoted_evaluation_can_plan_official_suite_after_temporal_gate(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "training.json"
    output = tmp_path / "evaluation.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_profile": "main",
                "train_size": 20000,
                "seeds": [17],
                "promoted_indices": [12],
                "baseline_ids": ["C3"],
                "target_name": "teacher-targets-tomato20000-v1.json",
                "target_gate_job_id": 950,
                "runs": [
                    {
                        "baseline_id": "C3",
                        "matrix_index": 12,
                        "seed": 17,
                        "backend": "distillm",
                        "training_dependency": "951",
                        "metadata_path": (
                            "checkpoints/C3-tomato20000-seed17-l896-4gpu/run_metadata.json"
                        ),
                        "model_path": "checkpoints/C3-tomato20000-seed17-l896-4gpu/5000",
                        "served_model_name": "Qwen/Qwen3-4B",
                        "model_revision": "revision",
                        "lora_path": None,
                        "model_dtype": "bfloat16",
                        "generation_config": "configs/generation/eval_qwen3_4b.yaml",
                        "evaluation_id": "C3-tomato20000-seed17-temporal-k16",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/submit_promoted_evaluations.py",
            "--manifest",
            str(manifest),
            "--teacher-score-job-id",
            "952",
            "--output",
            str(output),
            "--run-official",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    official = result["runs"][0]["official"]
    assert official["environment"]["MODEL_PATH"].endswith("/5000")
    assert official["environment"]["MODEL_DTYPE"] == "bfloat16"
    assert official["dependency_source"] == "temporal.evaluation_final"
    assert result["analysis"]["environment"]["RUN_OFFICIAL"] == "1"


def test_promoted_analysis_is_seed_balanced_artifact_audited_and_secondary_taste_aware() -> None:
    script = Path("slurm/analyze_promoted_matrix.sbatch").read_text(encoding="utf-8")

    assert "scripts/collect_seeded_evaluation_metrics.py" in script
    assert "--filter-absent-contrasts" in script
    assert "scripts/audit_training_matrix.py" in script
    assert '--baseline-id "${baseline_id}"' in script
    assert "training-matrix-audit-seed${seed}.json" in script
    assert "scripts/analyze_research_taste.py" in script
    assert "scripts/collect_seeded_official_metrics.py" in script
    assert "scripts/export_wandb_snapshot.py" in script


def test_replication_control_launcher_builds_independent_scored_taste_graph() -> None:
    script = Path("scripts/submit_replication_controls.sh").read_text(encoding="utf-8")

    assert "configs/generation/eval_qwen3_8b.yaml" in script
    assert "configs/generation/eval_qwen3_1p7b.yaml" in script
    assert 'teacher_id="A1-qwen8b-temporal-k16-seed17000"' in script
    assert 'student_id="A0-qwen1p7b-temporal-k16-seed17000"' in script
    assert "for index in 0 1" in script
    assert "slurm/sglang_smoke.sbatch" in script
    assert "slurm/score_teacher.sbatch" in script
    assert "slurm/annotate_research_taste.sbatch" in script
    assert script.count("slurm/evaluate_student.sbatch") >= 2
    assert '"teacher_score_final":"%s"' in script


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
