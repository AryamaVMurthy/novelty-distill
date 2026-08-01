import json
from pathlib import Path

import pytest

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.gem import (
    GEMRunSpec,
    build_gem_rows,
    build_official_gem_command,
    install_gem_trainer_compat,
    load_teacher_targets,
    tokenize_gem_example,
    write_gem_jsonl,
)


def test_gem_environment_pins_deepspeed_runtime_build_dependency() -> None:
    requirements = Path("environments/gem.in").read_text(encoding="utf-8").splitlines()

    assert "setuptools==83.0.0" in requirements


def test_gem_compat_accepts_new_transformers_learning_rate_argument() -> None:
    class OfficialTrainer:
        def _maybe_log_save_evaluate(
            self,
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
        ):
            return (tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

    assert install_gem_trainer_compat(OfficialTrainer) is True
    result = OfficialTrainer()._maybe_log_save_evaluate(
        1, 2, 3, 4, 5, 6, 7, learning_rate=8
    )

    assert result == (1, 2, 3, 4, 5, 6, 7)
    assert install_gem_trainer_compat(OfficialTrainer) is False


def test_gem_tokenization_masks_prompt_and_trains_only_on_completion() -> None:
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Can a coating improve stability?",
            "background_survey": "Humidity damages the catalyst.",
            "fine_grained_hypothesis": "A hydrophobic coating will help.",
            "inspiration": [],
        },
        split="train",
        task="open",
    )

    class FixedTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": len(messages) == 1,
                "enable_thinking": False,
            }
            return [10, 11] if len(messages) == 1 else [10, 11, 20, 21]

    row = tokenize_gem_example(
        example,
        target=example.human_target,
        tokenizer=FixedTokenizer(),
        max_length=8,
    )

    assert row == {
        "input_ids": [10, 11, 20, 21],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, -100, 20, 21],
    }


def test_gem_builds_exactly_four_teacher_rows_for_b4() -> None:
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Can a coating improve stability?",
            "background_survey": "Humidity damages the catalyst.",
            "fine_grained_hypothesis": "A hydrophobic coating will help.",
            "inspiration": [],
        },
        split="train",
        task="open",
    )
    baseline = next(
        item
        for item in load_baseline_registry(Path("configs/baselines.yaml")).baselines
        if item.id == "B4"
    )

    class TargetTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return [10, 11] if len(messages) == 1 else [10, 11, len(messages[1]["content"])]

    targets = ("one", "two", "three", "four")
    rows = build_gem_rows(
        baseline,
        (example,),
        teacher_targets={"paper-1": {"diverse4": targets}},
        tokenizer=TargetTokenizer(),
        max_length=16,
    )

    assert len(rows) == 4
    assert [row["labels"][-1] for row in rows] == [len(target) for target in targets]


def test_gem_rejects_incomplete_diverse_teacher_view() -> None:
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Question",
            "background_survey": "Background",
            "fine_grained_hypothesis": "Target",
            "inspiration": [],
        },
        split="train",
        task="open",
    )
    baseline = next(
        item
        for item in load_baseline_registry(Path("configs/baselines.yaml")).baselines
        if item.id == "B4"
    )

    with pytest.raises(ValueError, match="diverse4 requires 4 targets"):
        build_gem_rows(
            baseline,
            (example,),
            teacher_targets={"paper-1": {"diverse4": ("one",)}},
            tokenizer=object(),
            max_length=16,
        )


def test_teacher_target_artifact_and_gem_jsonl_are_plain_official_inputs(tmp_path: Path) -> None:
    targets_path = tmp_path / "teacher-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "targets": {"paper-1": {"diverse4": ["one", "two", "three", "four"]}},
            }
        ),
        encoding="utf-8",
    )
    targets = load_teacher_targets(targets_path)
    output = tmp_path / "gem.jsonl"
    rows = (
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [1, 3], "attention_mask": [1, 1], "labels": [-100, 3]},
    )

    write_gem_jsonl(rows, output)

    assert targets["paper-1"]["diverse4"] == ("one", "two", "three", "four")
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == list(
        rows
    )


def test_gem_command_invokes_pinned_official_entrypoint() -> None:
    spec = GEMRunSpec(
        baseline_id="B4",
        model="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        input=Path("data/input.jsonl"),
        teacher_targets=Path("data/teacher-targets.json"),
        tokenized_output=Path("data/gem.jsonl"),
        output_dir=Path("checkpoints/B4-gem-smoke"),
        max_examples=1,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        max_length=768,
        gem_beta=0.7,
        seed=17,
    )

    command = build_official_gem_command(
        spec,
        python_executable=Path("/env/bin/python"),
        official_checkout=Path("/official/gem"),
        model_path=Path("/models/qwen"),
        tokenized_path=Path("/data/gem.jsonl"),
        output_dir=Path("/output"),
    )

    assert command[:9] == (
        "/env/bin/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--module",
        "novelty_distill.training.gem_compat",
        "--official-train",
    )
    assert command[9] == "/official/gem/train.py"
    assert command[command.index("--loss") + 1] == "gem"
    assert command[command.index("--gem_beta") + 1] == "0.7"
    assert command[command.index("--model_name_or_path") + 1] == "/models/qwen"
