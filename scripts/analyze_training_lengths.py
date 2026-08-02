#!/usr/bin/env python3
"""Audit pinned-tokenizer context pressure before TOMATO training starts."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from novelty_distill.training.gem import load_teacher_targets
from novelty_distill.training.length_audit import summarize_token_lengths
from novelty_distill.training.opsd import render_privileged_prompt_pairs
from novelty_distill.training.trl import load_canonical_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--distillm-max-prompt-length", type=int, default=480)
    parser.add_argument("--distillm-max-length", type=int, default=896)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _chat_tokens(tokenizer: Any, *, prompt: str, target: str | None = None) -> int:
    messages = [{"role": "user", "content": prompt}]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=target is None,
        enable_thinking=False,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _longest(rows: Any, *, count: int = 10) -> list[dict[str, int | str]]:
    return [
        {"id": row_id, "length": length}
        for row_id, length in sorted(rows, key=lambda row: (-row[1], row[0]))[:count]
    ]


def main() -> None:
    args = parse_args()
    if args.max_examples <= 0 or args.max_new_tokens <= 0:
        raise ValueError("example and completion limits must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    examples = load_canonical_examples(args.input, limit=args.max_examples)
    teacher_targets = load_teacher_targets(args.teacher_targets)

    student_prompt_rows = tuple(
        (example.id, _chat_tokens(tokenizer, prompt=example.student_prompt))
        for example in examples
    )
    student_prompt_lengths = tuple(length for _id, length in student_prompt_rows)
    full_rows: dict[str, list[tuple[str, int]]] = {
        "human": [],
        "random1": [],
        "best1": [],
        "mode1": [],
        "diverse4": [],
    }
    for example in examples:
        views = {"human": (example.human_target,), **teacher_targets[example.id]}
        for view in full_rows:
            for target_index, target in enumerate(views[view]):
                full_rows[view].append(
                    (
                        f"{example.id}:{target_index}",
                        _chat_tokens(
                            tokenizer, prompt=example.student_prompt, target=target
                        ),
                    )
                )

    privileged_pairs = render_privileged_prompt_pairs(
        tuple(
            {
                "problem": example.student_prompt,
                "solution": (
                    "Historical hypothesis:\n"
                    f"{example.privileged_context.historical_hypothesis}\n\n"
                    "Known inspirations:\n"
                    + (
                        "\n".join(
                            f"{index}. {text}"
                            for index, text in enumerate(
                                example.privileged_context.inspirations, start=1
                            )
                        )
                        or "None supplied."
                    )
                ),
            }
            for example in examples
        ),
        tokenizer=tokenizer,
        student_thinking=False,
        teacher_thinking=False,
    )
    privileged_teacher_rows = tuple(
        (
            example.id,
            len(tokenizer(teacher, add_special_tokens=False)["input_ids"]),
        )
        for example, (_student, teacher) in zip(examples, privileged_pairs, strict=True)
    )
    privileged_teacher_lengths = tuple(
        length for _id, length in privileged_teacher_rows
    )
    full_lengths = {
        view: [length for _id, length in rows] for view, rows in full_rows.items()
    }

    summaries = {
        f"student_prompt_at_{args.max_length}": summarize_token_lengths(
            student_prompt_lengths, max_length=args.max_length
        ),
        f"privileged_teacher_prompt_at_{args.max_length}": summarize_token_lengths(
            privileged_teacher_lengths, max_length=args.max_length
        ),
        f"on_policy_prompt_plus_max_generation_at_{args.max_length}": summarize_token_lengths(
            (length + args.max_new_tokens for length in student_prompt_lengths),
            max_length=args.max_length,
        ),
        "distillm_prompt_proxy_at_480": summarize_token_lengths(
            student_prompt_lengths, max_length=args.distillm_max_prompt_length
        ),
        **{
            f"{view}_full_chat_at_{args.max_length}": summarize_token_lengths(
                lengths, max_length=args.max_length
            )
            for view, lengths in full_lengths.items()
        },
        "distillm_best1_full_chat_proxy_at_896": summarize_token_lengths(
            full_lengths["best1"], max_length=args.distillm_max_length
        ),
    }
    payload = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "num_examples": len(examples),
        "limits": {
            "trl_opsd_max_length": args.max_length,
            "on_policy_max_new_tokens": args.max_new_tokens,
            "distillm_max_prompt_length": args.distillm_max_prompt_length,
            "distillm_max_length": args.distillm_max_length,
        },
        "inputs": {
            "canonical_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "teacher_targets_sha256": hashlib.sha256(
                args.teacher_targets.read_bytes()
            ).hexdigest(),
        },
        "notes": {
            "on_policy": "upper bound from declared maximum generation, not observed truncation",
            "distillm": "Exact non-thinking Qwen chat template used by the task adapter",
        },
        "summaries": summaries,
        "longest_examples": {
            "student_prompt": _longest(student_prompt_rows),
            "privileged_teacher_prompt": _longest(privileged_teacher_rows),
            **{f"{view}_full_chat": _longest(rows) for view, rows in full_rows.items()},
        },
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
