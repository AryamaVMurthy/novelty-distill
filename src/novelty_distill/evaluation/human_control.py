"""Deterministic historical-target control under the model evaluation token budget."""

import hashlib
from typing import Any

from novelty_distill.data.tomato import CanonicalExample
from novelty_distill.generation.sglang import (
    GenerationRecord,
    GenerationSpec,
    generation_fingerprint,
    render_generation_prompt,
)


def build_human_control_record(
    example: CanonicalExample,
    *,
    spec: GenerationSpec,
    tokenizer: Any,
) -> GenerationRecord:
    """Represent one historical hypothesis as a bounded, auditable completion."""

    if spec.samples_per_prompt != 1:
        raise ValueError("historical control requires exactly one sample per prompt")
    full_token_ids = tokenizer.encode(example.human_target, add_special_tokens=False)
    token_ids = full_token_ids[: spec.max_new_tokens]
    if not token_ids:
        raise ValueError(f"historical target {example.id!r} tokenized to an empty sequence")
    truncated = len(full_token_ids) > len(token_ids)
    text = (
        tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if truncated
        else example.human_target.strip()
    )
    if not text:
        raise ValueError(f"historical target {example.id!r} decoded to empty text")
    prompt = render_generation_prompt(example.student_prompt, spec)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=spec.enable_thinking,
    )
    request_digest = hashlib.sha256(f"{example.id}\0{text}".encode()).hexdigest()[:24]
    return GenerationRecord(
        prompt_id=example.id,
        sample_index=0,
        text=text,
        finish_reason="length" if truncated else "stop",
        model=spec.model,
        request_id=f"historical-{request_digest}",
        config_hash=generation_fingerprint(spec),
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(token_ids),
    )
