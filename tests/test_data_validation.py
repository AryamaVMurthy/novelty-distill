import hashlib
from pathlib import Path

import pytest

from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.data.validation import validate_tomato_family


def _write(path: Path, *, ids: tuple[str, ...], task: str, split: str) -> None:
    rows = [
        prepare_tomato_record(
            {
                "source_id": source_id,
                "research_question": "Can a coating improve stability?",
                "background_survey": "Humidity damages the catalyst.",
                "fine_grained_hypothesis": "A hydrophobic coating will help.",
                "inspiration": [{"insp": "Use a porous protective shell."}],
            },
            split=split,
            task=task,
        )
        for source_id in ids
    ]
    path.write_text("".join(f"{row.model_dump_json()}\n" for row in rows))


def test_tomato_family_validation_proves_nested_paired_disjoint_artifacts(
    tmp_path: Path,
) -> None:
    open_small = tmp_path / "open-2.jsonl"
    open_large = tmp_path / "open-3.jsonl"
    composition_small = tmp_path / "composition-2.jsonl"
    composition_large = tmp_path / "composition-3.jsonl"
    test_open = tmp_path / "test-open.jsonl"
    test_composition = tmp_path / "test-composition.jsonl"
    _write(open_small, ids=("a", "b"), task="open", split="train")
    _write(open_large, ids=("a", "b", "c"), task="open", split="train")
    _write(composition_small, ids=("a", "b"), task="composition", split="train")
    _write(composition_large, ids=("a", "b", "c"), task="composition", split="train")
    _write(test_open, ids=("z",), task="open", split="test")
    _write(test_composition, ids=("z",), task="composition", split="test")

    manifest = validate_tomato_family(
        train_open={2: open_small, 3: open_large},
        train_composition={2: composition_small, 3: composition_large},
        test_open=test_open,
        test_composition=test_composition,
        expected_test_size=1,
    )

    assert manifest["train_sizes"] == [2, 3]
    assert manifest["test_size"] == 1
    assert manifest["artifacts"][str(open_small)]["sha256"] == hashlib.sha256(
        open_small.read_bytes()
    ).hexdigest()


def test_tomato_family_validation_rejects_non_nested_or_leaking_ids(
    tmp_path: Path,
) -> None:
    open_small = tmp_path / "open-1.jsonl"
    open_large = tmp_path / "open-2.jsonl"
    composition_small = tmp_path / "composition-1.jsonl"
    composition_large = tmp_path / "composition-2.jsonl"
    test_open = tmp_path / "test-open.jsonl"
    test_composition = tmp_path / "test-composition.jsonl"
    _write(open_small, ids=("not-nested",), task="open", split="train")
    _write(open_large, ids=("a", "leak"), task="open", split="train")
    _write(composition_small, ids=("not-nested",), task="composition", split="train")
    _write(composition_large, ids=("a", "leak"), task="composition", split="train")
    _write(test_open, ids=("leak",), task="open", split="test")
    _write(test_composition, ids=("leak",), task="composition", split="test")

    with pytest.raises(ValueError, match="nested"):
        validate_tomato_family(
            train_open={1: open_small, 2: open_large},
            train_composition={1: composition_small, 2: composition_large},
            test_open=test_open,
            test_composition=test_composition,
            expected_test_size=1,
        )
