from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.data.training_rows import to_chat_row, to_opsd_row


def test_training_rows_keep_privilege_out_of_student_context() -> None:
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "How can catalyst stability be improved?",
            "background_survey": "Catalysts deactivate under humid conditions.",
            "fine_grained_hypothesis": "A hydrophobic shell will slow deactivation.",
            "inspiration": [{"insp": "Hydrophobic confinement protects active sites."}],
        },
        split="train",
        task="open",
    )

    chat = to_chat_row(example, target="A teacher-generated hypothesis.")
    opsd = to_opsd_row(example)

    assert chat["messages"][-1] == {
        "role": "assistant",
        "content": "A teacher-generated hypothesis.",
    }
    assert "hydrophobic shell" not in chat["messages"][0]["content"].lower()
    assert opsd["problem"] == example.student_prompt
    assert "A hydrophobic shell" in opsd["solution"]
    assert "Hydrophobic confinement" in opsd["solution"]
