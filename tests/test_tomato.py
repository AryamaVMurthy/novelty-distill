import pytest

from novelty_distill.data.tomato import prepare_tomato_record, select_deterministic_subset


def test_open_task_separates_student_prompt_from_privileged_context() -> None:
    raw = {
        "source_id": "2025_12345678",
        "title": "A revealing paper title",
        "doi": "10.1000/secret",
        "research_question": "How can catalyst stability be improved?",
        "background_survey": "Existing catalysts deactivate under humid conditions.",
        "fine_grained_hypothesis": "A hydrophobic shell will slow catalyst deactivation.",
        "inspiration": [
            {
                "insp": "Hydrophobic confinement protects reactive sites.",
                "found_title": "Another revealing title",
                "found_doi": "10.1000/inspiration",
            }
        ],
    }

    example = prepare_tomato_record(raw, split="train", task="open")

    assert example.id == "2025_12345678"
    assert len(example.prompt_hash) == 64
    assert "How can catalyst stability be improved?" in example.student_prompt
    assert "Existing catalysts deactivate" in example.student_prompt
    assert "A revealing paper title" not in example.student_prompt
    assert "10.1000" not in example.student_prompt
    assert "Another revealing title" not in example.student_prompt
    assert example.human_target == "A hydrophobic shell will slow catalyst deactivation."
    assert example.privileged_context.historical_hypothesis == example.human_target
    assert example.privileged_context.inspirations == (
        "Hydrophobic confinement protects reactive sites.",
    )
    assert example.source_ids == ("2025_12345678",)
    assert example.split == "train"


def test_composition_task_exposes_inspiration_text_without_source_identity() -> None:
    raw = {
        "source_id": "2025_12345678",
        "title": "A revealing paper title",
        "doi": "10.1000/secret",
        "research_question": "How can catalyst stability be improved?",
        "background_survey": "Existing catalysts deactivate under humid conditions.",
        "fine_grained_hypothesis": "A hydrophobic shell will slow catalyst deactivation.",
        "inspiration": [
            {
                "insp": "Hydrophobic confinement protects reactive sites.",
                "found_title": "Another revealing title",
                "found_doi": "10.1000/inspiration",
            }
        ],
    }

    example = prepare_tomato_record(raw, split="train", task="composition")

    assert "Hydrophobic confinement protects reactive sites." in example.student_prompt
    assert "Another revealing title" not in example.student_prompt
    assert "10.1000" not in example.student_prompt
    assert example.task == "composition"


def test_composition_task_requires_at_least_one_inspiration() -> None:
    raw = {
        "source_id": "2025_12345678",
        "research_question": "How can catalyst stability be improved?",
        "background_survey": "Existing catalysts deactivate under humid conditions.",
        "fine_grained_hypothesis": "A hydrophobic shell will slow catalyst deactivation.",
        "inspiration": [],
    }

    with pytest.raises(ValueError, match="requires at least one inspiration"):
        prepare_tomato_record(raw, split="train", task="composition")


def test_fixed_subset_is_order_independent_and_nested() -> None:
    records = [{"source_id": f"paper-{index}"} for index in range(20)]

    small = select_deterministic_subset(records, size=5, seed=7)
    large = select_deterministic_subset(reversed(records), size=10, seed=7)

    assert [item["source_id"] for item in small] == [
        item["source_id"] for item in large[:5]
    ]
