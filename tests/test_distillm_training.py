from pathlib import Path

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.distillm import (
    DistiLLMRunSpec,
    build_distillm_preprocess_command,
    build_distillm_raw_rows,
    build_distillm_training_command,
    normalize_distillm_qwen_sentinels,
)


def test_distillm_environment_pins_deepspeed_runtime_build_dependency() -> None:
    requirements = Path("environments/distillm.in").read_text(
        encoding="utf-8"
    ).splitlines()

    assert "setuptools==83.0.0" in requirements


def test_qwen_uint32_separator_is_normalized_for_official_loader(tmp_path: Path) -> None:
    data_path = tmp_path / "train_0.bin"
    data_path.write_bytes(
        b"".join(value.to_bytes(4, "little") for value in (1, 2**32 - 1, 2))
    )

    assert normalize_distillm_qwen_sentinels(tmp_path) == 1
    payload = data_path.read_bytes()
    assert [
        int.from_bytes(payload[offset : offset + 4], "little")
        for offset in range(0, len(payload), 4)
    ] == [1, 65535, 2]
    assert normalize_distillm_qwen_sentinels(tmp_path) == 0


def _spec() -> DistiLLMRunSpec:
    return DistiLLMRunSpec(
        baseline_id="C3",
        student_model="Qwen/Qwen3-1.7B",
        student_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        teacher_model="Qwen/Qwen3-14B",
        teacher_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        input=Path("data/input.jsonl"),
        teacher_targets=Path("data/teacher-targets.json"),
        raw_dir=Path("data/distillm/raw"),
        processed_dir=Path("data/distillm/processed"),
        output_dir=Path("checkpoints/C3-distillm-smoke"),
        max_examples=2,
        dev_examples=1,
        max_steps=1,
        batch_size=1,
        learning_rate=5e-6,
        max_length=256,
        max_prompt_length=192,
        skew_alpha=0.1,
        seed=17,
    )


def test_distillm_raw_rows_use_static_best_teacher_target() -> None:
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Question",
            "background_survey": "Background",
            "fine_grained_hypothesis": "Historical target",
            "inspiration": [],
        },
        split="train",
        task="open",
    )
    baseline = next(
        item
        for item in load_baseline_registry(Path("configs/baselines.yaml")).baselines
        if item.id == "C3"
    )

    rows = build_distillm_raw_rows(
        baseline,
        (example,),
        teacher_targets={"paper-1": {"best1": ("Teacher target",)}},
    )

    assert rows == (
        {"instruction": example.student_prompt, "input": "", "output": "Teacher target"},
    )


def test_distillm_commands_delegate_preprocessing_and_training_to_official_repo() -> None:
    spec = _spec()
    preprocess = build_distillm_preprocess_command(
        spec,
        python_executable=Path("/env/bin/python"),
        official_checkout=Path("/official/distillm"),
        student_model_path=Path("/models/student"),
        raw_dir=Path("/data/raw"),
        processed_dir=Path("/data/processed"),
    )
    training = build_distillm_training_command(
        spec,
        python_executable=Path("/env/bin/python"),
        official_checkout=Path("/official/distillm"),
        student_model_path=Path("/models/student"),
        teacher_model_path=Path("/models/teacher"),
        processed_dir=Path("/data/processed/qwen"),
        output_dir=Path("/output"),
    )

    assert preprocess[1] == "/official/distillm/tools/process_data_dolly.py"
    assert preprocess[preprocess.index("--model-type") + 1] == "qwen"
    assert training[:7] == (
        "/env/bin/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        "/official/distillm/finetune.py",
    )
    assert training[training.index("--type") + 1] == "adaptive-sfkl"
    assert training[training.index("--data-dir") + 1] == "/data/processed/qwen/"
    assert training[training.index("--deepspeed_config") + 1] == (
        "/official/distillm/configs/deepspeed/ds_config_zero2_offload.json"
    )
    assert "--student-gen" in training
    assert training[training.index("--skew-alpha") + 1] == "0.1"


def test_distillm_command_partitions_training_across_requested_gpus() -> None:
    spec = _spec().model_copy(update={"num_gpus": 2})

    training = build_distillm_training_command(
        spec,
        python_executable=Path("/env/bin/python"),
        official_checkout=Path("/official/distillm"),
        student_model_path=Path("/models/student"),
        teacher_model_path=Path("/models/teacher"),
        processed_dir=Path("/data/processed/qwen"),
        output_dir=Path("/output"),
    )

    assert training[training.index("--nproc_per_node=2")] == "--nproc_per_node=2"
    assert training[training.index("--n-gpu") + 1] == "2"
