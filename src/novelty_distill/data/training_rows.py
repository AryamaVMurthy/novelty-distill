"""Minimal rows consumed by the official TRL and OPSD data collators."""

from typing import TypedDict

from novelty_distill.data.tomato import CanonicalExample


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatTrainingRow(TypedDict):
    id: str
    messages: list[ChatMessage]


class OPSDTrainingRow(TypedDict):
    id: str
    problem: str
    solution: str


def to_chat_row(example: CanonicalExample, *, target: str) -> ChatTrainingRow:
    """Build the conversational row expected by TRL SFT/GKD collators."""

    stripped_target = target.strip()
    if not stripped_target:
        raise ValueError("training target cannot be empty")
    return {
        "id": example.id,
        "messages": [
            {"role": "user", "content": example.student_prompt},
            {"role": "assistant", "content": stripped_target},
        ],
    }


def to_opsd_row(example: CanonicalExample) -> OPSDTrainingRow:
    """Map TOMATO privilege into the official OPSD `problem`/`solution` schema."""

    inspiration_text = "\n".join(
        f"{index}. {inspiration}"
        for index, inspiration in enumerate(
            example.privileged_context.inspirations,
            start=1,
        )
    )
    solution = (
        "Historical hypothesis:\n"
        f"{example.privileged_context.historical_hypothesis}\n\n"
        "Known inspirations:\n"
        f"{inspiration_text or 'None supplied.'}"
    )
    return {"id": example.id, "problem": example.student_prompt, "solution": solution}
