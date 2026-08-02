from pathlib import Path

import pytest
import yaml

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.distillm import (
    DistiLLMRunSpec,
    _latest_distillm_checkpoint,
    audit_distillm_log,
    build_distillm_raw_rows,
    build_distillm_training_command,
    distillm_epoch_plan,
    encode_distillm_chat_row,
    normalize_distillm_qwen_sentinels,
)


def test_distillm_chat_adapter_uses_qwen_template_and_preserves_prompt() -> None:
    class CharacterTokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            assert kwargs["enable_thinking"] is False
            return "PP" if len(messages) == 1 else "PPabcdefgh"

        def __call__(self, text, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": [ord(character) for character in text]}

    encoded = encode_distillm_chat_row(
        {"instruction": "problem", "input": "", "output": "answer"},
        tokenizer=CharacterTokenizer(),
        max_length=6,
        max_prompt_length=4,
    )

    assert encoded["prompt_ids"] == [ord("P"), ord("P")]
    assert encoded["completion_ids"] == [ord(character) for character in "abcd"]
    assert encoded["truncated_completion_tokens"] == 4


def test_distillm_chat_adapter_rejects_legacy_separator_token_collision() -> None:
    class CollidingTokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            return "prompt" if len(messages) == 1 else "promptanswer"

        def __call__(self, text, **kwargs):
            return {"input_ids": [65535] if text == "prompt" else [65535, 1]}

    with pytest.raises(ValueError, match="separator token"):
        encode_distillm_chat_row(
            {"instruction": "problem", "input": "", "output": "answer"},
            tokenizer=CollidingTokenizer(),
            max_length=6,
            max_prompt_length=4,
        )


def test_distillm_log_audit_records_adaptive_threshold_trajectory(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "log.txt"
    log_path.write_text(
        "dev | avg_loss: 1.0 | {} | threshold: 0.0\n"
        "train | epoch   0 | Iter:      1/     2 | global iter:      1/     2 | "
        "loss: 0.5 | ds_loss: 0.5 | lr: 5e-6 | scale: 1 | micro time: 1 | step time: 1\n"
        "dev | avg_loss: 1.2 | {} | threshold: 0.0\n"
        "train | epoch   1 | Iter:      2/     2 | global iter:      2/     2 | "
        "loss: 0.4 | ds_loss: 0.4 | lr: 5e-6 | scale: 1 | micro time: 1 | step time: 1\n"
        "dev | avg_loss: 1.1 | {} | threshold: 0.1\n",
        encoding="utf-8",
    )

    assert audit_distillm_log(log_path, expected_steps=2) == {
        "logged_training_steps": 2,
        "last_global_step": 2,
        "validation_checks": 3,
        "validation_losses": [1.0, 1.2, 1.1],
        "adaptive_thresholds": [0.0, 0.0, 0.1],
        "terminal_adaptive_threshold": 0.1,
    }


def test_distillm_log_audit_ignores_preserved_previous_invocations(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "log.txt"
    old_log = "dev | avg_loss: 99.0 | {} | threshold: 0.9\n"
    log_path.write_text(
        old_log
        + "dev | avg_loss: 1.0 | {} | threshold: 0.0\n"
        + "train | global iter: 1/1 | loss: 0.5\n",
        encoding="utf-8",
    )

    audit = audit_distillm_log(
        log_path, expected_steps=1, start_offset=len(old_log.encode())
    )

    assert audit["validation_losses"] == [1.0]
    assert audit["adaptive_thresholds"] == [0.0]


def test_distillm_environment_pins_deepspeed_runtime_build_dependency() -> None:
    requirements = Path("environments/distillm.in").read_text(
        encoding="utf-8"
    ).splitlines()

    assert "setuptools==83.0.0" in requirements


def test_distillm_small_smoke_fits_task_faithful_first_prompt() -> None:
    spec = DistiLLMRunSpec.model_validate(
        yaml.safe_load(
            Path("configs/training/distillm_smoke.yaml").read_text(encoding="utf-8")
        )
    )

    assert spec.max_prompt_length == 256
    assert spec.max_length == 512


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
        validation_interval=2,
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


def test_distillm_training_command_delegates_to_official_repo() -> None:
    spec = _spec()
    training = build_distillm_training_command(
        spec,
        python_executable=Path("/env/bin/python"),
        official_checkout=Path("/official/distillm"),
        student_model_path=Path("/models/student"),
        teacher_model_path=Path("/models/teacher"),
        processed_dir=Path("/data/processed/qwen"),
        output_dir=Path("/output"),
    )

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
    spec = _spec().model_copy(update={"num_gpus": 2, "max_examples": 3})

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


def test_distillm_production_repeats_only_enough_data_to_reach_save_step() -> None:
    spec = DistiLLMRunSpec.model_validate(
        yaml.safe_load(
            Path("configs/training/distillm_tomato1k.yaml").read_text(encoding="utf-8")
        )
    )

    plan = distillm_epoch_plan(spec)
    training = build_distillm_training_command(
        spec,
        python_executable=Path("/env/bin/python"),
        official_checkout=Path("/official/distillm"),
        student_model_path=Path("/models/student"),
        teacher_model_path=Path("/models/teacher"),
        processed_dir=Path("/data/processed/qwen"),
        output_dir=Path("/output"),
    )

    assert plan == {"train_examples": 960, "steps_per_epoch": 240, "epochs": 2}
    assert training[training.index("--epochs") + 1] == "2"
    assert training[training.index("--eval-interval") + 1] == "25"
    assert spec.max_steps * spec.batch_size * spec.num_gpus == 1000


def test_distillm_spec_rejects_too_few_rows_for_distributed_batch() -> None:
    payload = _spec().model_dump()
    payload.update({"num_gpus": 2, "max_examples": 2, "dev_examples": 1})

    with pytest.raises(ValueError, match="distributed batch"):
        DistiLLMRunSpec.model_validate(payload)


def test_latest_distillm_checkpoint_requires_deployable_model_files(tmp_path: Path) -> None:
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "1" / "pytorch_model.bin").write_bytes(b"weights")
    (tmp_path / "2").mkdir()
    (tmp_path / "2" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "2" / "model.safetensors").write_bytes(b"weights")

    assert _latest_distillm_checkpoint(tmp_path) == tmp_path / "2"


def test_latest_distillm_checkpoint_fails_closed_without_weights(tmp_path: Path) -> None:
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="deployable checkpoint"):
        _latest_distillm_checkpoint(tmp_path)
