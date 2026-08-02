from novelty_distill.data.ultrafeedback import (
    prepare_ultrafeedback_record,
    select_unique_content_indices,
    ultrafeedback_record_id,
    ultrafeedback_record_is_usable,
)


def _completion(model: str, response: str, score: float) -> dict[str, object]:
    return {
        "model": model,
        "response": response,
        "overall_score": score,
        "fine-grained_score": score / 2,
    }


def test_ultrafeedback_record_derives_four_reusable_views() -> None:
    raw = {
        "source": "test-source",
        "instruction": "Explain the result.",
        "models": ["model-z", "model-a", "model-c", "model-b"],
        "completions": [
            _completion("model-z", "alpha beta gamma", 3),
            _completion("model-a", "alpha beta", 9),
            _completion("model-c", "red green blue", 7),
            _completion("model-b", "alpha beta delta", 5),
        ],
    }

    example, targets = prepare_ultrafeedback_record(raw, seed=17)

    assert example.student_prompt == "Explain the result."
    assert example.human_target == "alpha beta"
    assert example.dataset_revision == "40b436560ca83a8dba36114c22ab3c66e43f6d5e"
    assert targets["best1"] == ["alpha beta"]
    assert targets["mode1"] == targets["best1"]
    assert targets["diverse4"] == [
        "alpha beta",
        "alpha beta delta",
        "red green blue",
        "alpha beta gamma",
    ]
    assert targets["diverse2"] == ["alpha beta", "red green blue"]
    assert len(targets["random1"]) == 1
    assert targets["all4"] == targets["diverse4"]


def test_ultrafeedback_views_are_independent_of_completion_input_order() -> None:
    completions = [
        _completion("model-z", "alpha beta gamma", 3),
        _completion("model-a", "alpha beta", 9),
        _completion("model-c", "red green blue", 7),
        _completion("model-b", "alpha beta delta", 5),
    ]
    base = {
        "source": "test-source",
        "instruction": "Explain the result.",
        "models": ["model-z", "model-a", "model-c", "model-b"],
    }

    _, first = prepare_ultrafeedback_record({**base, "completions": completions}, seed=17)
    _, second = prepare_ultrafeedback_record(
        {**base, "completions": list(reversed(completions))}, seed=17
    )

    assert first == second


def test_content_identity_distinguishes_duplicate_instructions_and_ignores_order() -> None:
    completions = [
        _completion("model-z", "alpha beta gamma", 3),
        _completion("model-a", "alpha beta", 9),
        _completion("model-c", "red green blue", 7),
        _completion("model-b", "alpha beta delta", 5),
    ]
    raw = {
        "source": "test-source",
        "instruction": "Repeated instruction.",
        "models": ["model-z", "model-a", "model-c", "model-b"],
        "completions": completions,
    }
    changed = {
        **raw,
        "completions": [
            {**completions[0], "response": "different response"},
            *completions[1:],
        ],
    }

    assert ultrafeedback_record_id(raw) == ultrafeedback_record_id(
        {**raw, "completions": list(reversed(completions))}
    )
    assert ultrafeedback_record_id(raw) != ultrafeedback_record_id(changed)


def test_pilot_selection_deduplicates_exact_records_before_subsetting() -> None:
    identities = ("duplicate", "unique-b", "duplicate", "unique-a")

    selected = select_unique_content_indices(identities, size=3, seed=17)

    assert len(selected) == 3
    assert {identities[index] for index in selected} == {
        "duplicate",
        "unique-a",
        "unique-b",
    }
    assert 2 not in selected


def test_pilot_filter_rejects_empty_official_completions() -> None:
    raw = {
        "source": "test-source",
        "instruction": "Explain the result.",
        "models": ["model-z", "model-a", "model-c", "model-b"],
        "completions": [
            _completion("model-z", "", 3),
            _completion("model-a", "alpha beta", 9),
            _completion("model-c", "red green blue", 7),
            _completion("model-b", "alpha beta delta", 5),
        ],
    }

    assert not ultrafeedback_record_is_usable(raw)
