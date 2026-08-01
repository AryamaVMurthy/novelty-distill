"""Generation backends."""

from novelty_distill.generation.sglang import GenerationSpec, build_chat_completion_payload

__all__ = ["GenerationSpec", "build_chat_completion_payload"]
