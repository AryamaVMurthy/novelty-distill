"""Tokenized-data adapter for the pinned official GEM trainer."""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from novelty_distill.config import BaselineConfig
from novelty_distill.data.tomato import CanonicalExample


class GEMTokenizedRow(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


TeacherTargets = Mapping[str, Mapping[str, Sequence[str]]]


def load_teacher_targets(path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """Load the versioned target-view artifact shared by teacher-trained baselines."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported teacher-target schema in {path}")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, Mapping):
        raise ValueError(f"teacher-target artifact has no target mapping: {path}")

    targets: dict[str, dict[str, tuple[str, ...]]] = {}
    for prompt_id, raw_views in raw_targets.items():
        if not isinstance(prompt_id, str) or not prompt_id or not isinstance(raw_views, Mapping):
            raise ValueError(f"invalid teacher-target prompt entry in {path}")
        views: dict[str, tuple[str, ...]] = {}
        for view, raw_values in raw_views.items():
            if (
                not isinstance(view, str)
                or not isinstance(raw_values, list)
                or not raw_values
                or any(not isinstance(value, str) or not value.strip() for value in raw_values)
            ):
                raise ValueError(f"invalid teacher target view {view!r} for {prompt_id}")
            views[view] = tuple(value.strip() for value in raw_values)
        targets[prompt_id] = views
    return targets


def write_gem_jsonl(rows: Sequence[GEMTokenizedRow], path: Path) -> None:
    """Atomically write plain JSONL accepted by official GEM's `load_dataset('json')`."""

    if not rows:
        raise ValueError("GEM training data cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                json.dump(row, handle, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_gem_rows(
    baseline: BaselineConfig,
    examples: Sequence[CanonicalExample],
    *,
    teacher_targets: TeacherTargets,
    tokenizer: Any,
    max_length: int,
) -> tuple[GEMTokenizedRow, ...]:
    """Build the pretokenized rows consumed unchanged by official GEM `train.py`."""

    if baseline.backend != "gem":
        raise ValueError(f"baseline {baseline.id} does not use GEM")
    if baseline.trajectory_source != "teacher" or baseline.target_view != "diverse4":
        raise ValueError(f"baseline {baseline.id} does not declare teacher diverse4 targets")

    rows: list[GEMTokenizedRow] = []
    for example in examples:
        try:
            targets = tuple(teacher_targets[example.id][baseline.target_view])
        except KeyError as error:
            raise ValueError(f"missing diverse4 teacher targets for {example.id}") from error
        if len(targets) != 4:
            raise ValueError(f"diverse4 requires 4 targets for {example.id}")
        rows.extend(
            tokenize_gem_example(
                example,
                target=target,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            for target in targets
        )
    return tuple(rows)


def tokenize_gem_example(
    example: CanonicalExample,
    *,
    target: str,
    tokenizer: Any,
    max_length: int,
) -> GEMTokenizedRow:
    """Build the `input_ids`/`labels` row required by official GEM `train.py`."""

    prompt_messages = [{"role": "user", "content": example.student_prompt}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": target.strip()},
    ]
    prompt_ids = list(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    input_ids = list(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full training sequence")
    if len(input_ids) > max_length:
        raise ValueError(
            f"tokenized example {example.id} has {len(input_ids)} tokens, above {max_length}"
        )
    if len(input_ids) == len(prompt_ids):
        raise ValueError(f"tokenized example {example.id} has no completion tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }
