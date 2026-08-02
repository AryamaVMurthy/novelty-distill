"""Fail-closed validation for the canonical TOMATO dataset family."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from novelty_distill.data.tomato import TOMATO_REVISION, CanonicalExample


def _load_artifact(
    path: Path,
    *,
    expected_size: int,
    expected_split: Literal["train", "test"],
    expected_task: Literal["open", "composition"],
) -> tuple[CanonicalExample, ...]:
    rows: list[CanonicalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = CanonicalExample.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"invalid canonical row at {path}:{line_number}") from error
            if row.split != expected_split or row.task != expected_task:
                raise ValueError(f"unexpected split or task at {path}:{line_number}")
            if row.dataset_revision != TOMATO_REVISION:
                raise ValueError(f"unexpected dataset revision at {path}:{line_number}")
            if row.prompt_hash != hashlib.sha256(row.student_prompt.encode()).hexdigest():
                raise ValueError(f"prompt hash mismatch at {path}:{line_number}")
            rows.append(row)
    if len(rows) != expected_size:
        raise ValueError(f"expected {expected_size} rows in {path}, found {len(rows)}")
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {path}")
    return tuple(rows)


def validate_tomato_family(
    *,
    train_open: Mapping[int, Path],
    train_composition: Mapping[int, Path],
    test_open: Path,
    test_composition: Path,
    expected_test_size: int,
) -> dict[str, Any]:
    """Validate paired tasks, nested train IDs, and an untouched disjoint test split."""

    sizes = sorted(train_open)
    if not sizes or sizes != sorted(train_composition) or any(size <= 0 for size in sizes):
        raise ValueError("open and composition train artifacts need the same positive sizes")

    artifacts: dict[str, dict[str, Any]] = {}
    open_ids: dict[int, set[str]] = {}
    composition_ids: dict[int, set[str]] = {}
    for size in sizes:
        open_rows = _load_artifact(
            train_open[size],
            expected_size=size,
            expected_split="train",
            expected_task="open",
        )
        composition_rows = _load_artifact(
            train_composition[size],
            expected_size=size,
            expected_split="train",
            expected_task="composition",
        )
        open_ids[size] = {row.id for row in open_rows}
        composition_ids[size] = {row.id for row in composition_rows}
        if open_ids[size] != composition_ids[size]:
            raise ValueError(f"open and composition IDs differ at train size {size}")
        for path in (train_open[size], train_composition[size]):
            artifacts[str(path)] = {
                "rows": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    for smaller, larger in zip(sizes, sizes[1:], strict=False):
        if not open_ids[smaller] < open_ids[larger]:
            raise ValueError(f"train IDs are not strictly nested from {smaller} to {larger}")

    test_open_rows = _load_artifact(
        test_open,
        expected_size=expected_test_size,
        expected_split="test",
        expected_task="open",
    )
    test_composition_rows = _load_artifact(
        test_composition,
        expected_size=expected_test_size,
        expected_split="test",
        expected_task="composition",
    )
    test_ids = {row.id for row in test_open_rows}
    if test_ids != {row.id for row in test_composition_rows}:
        raise ValueError("open and composition test IDs differ")
    if test_ids & open_ids[sizes[-1]]:
        raise ValueError("train/test source ID leakage detected")
    for path in (test_open, test_composition):
        artifacts[str(path)] = {
            "rows": expected_test_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    return {
        "schema_version": 1,
        "dataset_revision": TOMATO_REVISION,
        "train_sizes": sizes,
        "test_size": expected_test_size,
        "artifacts": artifacts,
    }
