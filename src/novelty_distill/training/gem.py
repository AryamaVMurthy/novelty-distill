"""Tokenized-data adapter for the pinned official GEM trainer."""

from typing import Any, TypedDict

from novelty_distill.data.tomato import CanonicalExample


class GEMTokenizedRow(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


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
