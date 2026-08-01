from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.gem import tokenize_gem_example


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
