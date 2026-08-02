from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.evaluation.human_control import build_human_control_record
from novelty_distill.generation.sglang import GenerationSpec


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return " ".join(f"token-{value}" for value in token_ids)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
        assert kwargs["tokenize"] is True
        return list(range(len(messages[0]["content"].split()) + 2))


def test_human_control_record_applies_the_same_token_budget() -> None:
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Question",
            "background_survey": "Background",
            "fine_grained_hypothesis": "one two three four five",
            "inspiration": [],
        },
        split="test",
        task="open",
    )
    spec = GenerationSpec(
        model="historical-human-target",
        revision="fcd201d92758a642465a7653b8055f3a04d5f439",
        temperature=0,
        top_p=1,
        top_k=1,
        min_p=0,
        max_new_tokens=3,
        samples_per_prompt=1,
        seed=0,
        response_instruction="Return one answer.",
    )

    record = build_human_control_record(example, spec=spec, tokenizer=FakeTokenizer())

    assert record.text == "token-0 token-1 token-2"
    assert record.finish_reason == "length"
    assert record.completion_tokens == 3
    assert record.prompt_tokens is not None
    assert record.request_id.startswith("historical-")
