import json
from pathlib import Path

import pytest

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.gem import (
    build_gem_rows,
    load_teacher_targets,
    tokenize_gem_example,
    write_gem_jsonl,
)


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
