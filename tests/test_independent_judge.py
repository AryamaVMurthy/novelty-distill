import copy

import pytest

from novelty_distill.evaluation.independent_judge import (
    ExternalJudgeScore,
    analyze_independent_judgments,
    build_blinded_calibration_sample,
    build_independent_judge_payload,
    parse_independent_judge_response,
)


def _candidates(method: str, count: int = 12) -> list[dict]:
    return [
        {
            "prompt_id": f"p-{index:02d}",
            "sample_index": index % 4,
            "text": f"candidate response {index}",
            "dimensions": {
                "relevance": 3 + index % 3,
                "feasibility": 1 + index % 5,
                "soundness": 1 + (index // 2) % 5,
                "clarity": 4,
                "instruction_compliance": 5,
            },
        }
        for index in range(count)
    ]


def test_calibration_sample_is_balanced_blinded_and_repeatable() -> None:
    candidates = {"A0": _candidates("A0"), "C1": _candidates("C1")}
    prompts = {f"p-{index:02d}": f"research prompt {index}" for index in range(12)}

    first = build_blinded_calibration_sample(
        candidates_by_method=candidates,
        prompts=prompts,
        samples_per_method=8,
        repeat_fraction=0.25,
        seed=17,
    )
    second = build_blinded_calibration_sample(
        candidates_by_method=copy.deepcopy(candidates),
        prompts=prompts,
        samples_per_method=8,
        repeat_fraction=0.25,
        seed=17,
    )

    assert first == second
    assert len(first) == 20
    originals = [entry for entry in first if entry.repeat_of is None]
    repeats = [entry for entry in first if entry.repeat_of is not None]
    assert {method: sum(item.method == method for item in originals) for method in candidates} == {
        "A0": 8,
        "C1": 8,
    }
    assert {entry.score_stratum for entry in originals} == {0, 1, 2, 3}
    assert len(repeats) == 4
    assert len({entry.blind_id for entry in first}) == len(first)

    payload = build_independent_judge_payload(first[0], model="independent/model")
    serialized = str(payload)
    assert first[0].method not in serialized
    assert "score_stratum" not in serialized
    assert "qwen_dimensions" not in serialized
    assert payload["temperature"] == 0
    assert payload["response_format"]["type"] == "json_schema"


def test_calibration_sample_rejects_missing_prompt_and_text_duplicates() -> None:
    candidates = {"A0": _candidates("A0", 8)}
    prompts = {f"p-{index:02d}": f"prompt {index}" for index in range(7)}
    with pytest.raises(ValueError, match="absent from prompts"):
        build_blinded_calibration_sample(
            candidates_by_method=candidates,
            prompts=prompts,
            samples_per_method=8,
            repeat_fraction=0,
            seed=17,
        )

    candidates["A0"][1]["prompt_id"] = candidates["A0"][0]["prompt_id"]
    candidates["A0"][1]["sample_index"] = candidates["A0"][0]["sample_index"]
    with pytest.raises(ValueError, match="duplicate candidate"):
        build_blinded_calibration_sample(
            candidates_by_method=candidates,
            prompts={f"p-{index:02d}": f"prompt {index}" for index in range(8)},
            samples_per_method=8,
            repeat_fraction=0,
            seed=17,
        )


def test_parse_and_analyze_independent_judgments() -> None:
    entries = build_blinded_calibration_sample(
        candidates_by_method={"A0": _candidates("A0", 8)},
        prompts={f"p-{index:02d}": f"prompt {index}" for index in range(8)},
        samples_per_method=8,
        repeat_fraction=0.25,
        seed=23,
    )
    body = {
        "id": "req-1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "choices": [
            {
                "message": {
                    "content": (
                        '{"relevance":4,"feasibility":3,"soundness":4,"clarity":4,'
                        '"instruction_compliance":5,"fatal_flaw":false,'
                        '"brief_rationale":"Actionable but one control is underspecified."}'
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    }
    parsed = parse_independent_judge_response(body)
    assert parsed.request_id == "req-1"
    assert parsed.score.feasibility == 3
    assert parsed.usage == {"prompt_tokens": 100, "completion_tokens": 30}

    ratings = {
        entry.blind_id: ExternalJudgeScore(
            relevance=entry.qwen_dimensions["relevance"],
            feasibility=entry.qwen_dimensions["feasibility"],
            soundness=entry.qwen_dimensions["soundness"],
            clarity=entry.qwen_dimensions["clarity"],
            instruction_compliance=entry.qwen_dimensions["instruction_compliance"],
            fatal_flaw=False,
            brief_rationale="fixture",
        )
        for entry in entries
    }
    summary = analyze_independent_judgments(entries=entries, ratings=ratings)
    assert summary["counts"] == {"originals": 8, "repeats": 2, "total": 10}
    assert summary["agreement"]["feasibility"]["exact_rate"] == 1
    assert summary["agreement"]["soundness"]["pearson"] == 1
    assert summary["repeat_reliability"]["feasibility"]["mean_absolute_difference"] == 0
    assert summary["method_means"]["A0"]["feasibility"] == pytest.approx(
        sum(entry.qwen_dimensions["feasibility"] for entry in entries if entry.repeat_of is None)
        / 8
    )
    assert summary["paired_contrasts"] == {}
