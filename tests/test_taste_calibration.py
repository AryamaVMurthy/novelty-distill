import pytest

from novelty_distill.evaluation.taste_calibration import (
    analyze_human_taste_agreement,
    cohen_kappa,
    select_blinded_taste_calibration,
)


def _candidate(method: str, index: int) -> dict[str, object]:
    return {
        "method": method,
        "prompt_id": f"prompt-{method}-{index}",
        "sample_index": index,
        "task": f"Task {index}",
        "text": f"Candidate idea {method} {index}",
        "opportunity_pattern": "explanation_gap" if index % 2 else "failure_or_risk_gap",
        "method_paradigm": (
            "formal_conceptual_derivation"
            if index % 2
            else "failure_mitigation_or_robustification"
        ),
    }


def test_blinded_calibration_selection_is_deterministic_and_balanced() -> None:
    candidates = tuple(
        _candidate(method, index)
        for method in ("A3", "A1", "A0", "B1", "B3")
        for index in range(12)
    )

    first = select_blinded_taste_calibration(candidates, size=20, seed=17)
    second = select_blinded_taste_calibration(candidates, size=20, seed=17)

    assert first == second
    assert len(first["packet"]) == 20
    assert len(first["key"]) == 20
    assert first["stratum_counts"] == {"human": 5, "teacher": 5, "base": 5, "trained": 5}
    assert {record["method"] for record in first["key"] if record["stratum"] == "trained"} == {
        "B1",
        "B3",
    }
    assert all("method" not in record and "prompt_id" not in record for record in first["packet"])
    assert {record["calibration_id"] for record in first["packet"]} == {
        record["calibration_id"] for record in first["key"]
    }


def test_blinded_calibration_rejects_an_underfilled_stratum() -> None:
    candidates = tuple(
        _candidate(method, index)
        for method in ("A3", "A1", "A0", "B1")
        for index in range(2)
    )

    with pytest.raises(ValueError, match="human calibration stratum"):
        select_blinded_taste_calibration(candidates, size=12, seed=17)


def test_cohen_kappa_handles_perfect_and_chance_level_agreement() -> None:
    assert cohen_kappa(("a", "b"), ("a", "b"), categories=("a", "b")) == 1.0
    assert cohen_kappa(("a", "b"), ("a", "a"), categories=("a", "b")) == 0.0


def test_human_agreement_gate_requires_every_annotator_and_axis() -> None:
    key = tuple(
        {
            "calibration_id": f"c{index}",
            "opportunity_pattern": "explanation_gap" if index % 2 else "failure_or_risk_gap",
            "method_paradigm": (
                "formal_conceptual_derivation"
                if index % 2
                else "failure_mitigation_or_robustification"
            ),
        }
        for index in range(20)
    )
    human = {
        name: tuple(
            {
                "calibration_id": record["calibration_id"],
                "opportunity_pattern": record["opportunity_pattern"],
                "method_paradigm": record["method_paradigm"],
            }
            for record in key
        )
        for name in ("annotator-1", "annotator-2")
    }

    result = analyze_human_taste_agreement(
        key=key,
        humans=human,
        minimum_records=20,
        minimum_annotators=2,
        minimum_kappa=0.8,
    )

    assert result["passed"] is True
    assert result["automated_vs_human"]["opportunity_pattern"] == {
        "annotator-1": 1.0,
        "annotator-2": 1.0,
    }

    failing = dict(human)
    failing["annotator-2"] = tuple(
        {
            **record,
            "opportunity_pattern": (
                "failure_or_risk_gap"
                if record["opportunity_pattern"] == "explanation_gap"
                else "explanation_gap"
            ),
        }
        for record in human["annotator-2"]
    )
    failed = analyze_human_taste_agreement(
        key=key,
        humans=failing,
        minimum_records=20,
        minimum_annotators=2,
        minimum_kappa=0.8,
    )
    assert failed["passed"] is False
    assert "automated_vs_human.opportunity_pattern.annotator-2" in failed["failures"]


def test_human_agreement_rejects_missing_or_extra_calibration_ids() -> None:
    key = (
        {
            "calibration_id": "c1",
            "opportunity_pattern": "explanation_gap",
            "method_paradigm": "formal_conceptual_derivation",
        },
    )
    human = {
        "annotator-1": (
            {
                "calibration_id": "different",
                "opportunity_pattern": "explanation_gap",
                "method_paradigm": "formal_conceptual_derivation",
            },
        ),
        "annotator-2": (
            {
                "calibration_id": "c1",
                "opportunity_pattern": "explanation_gap",
                "method_paradigm": "formal_conceptual_derivation",
            },
        ),
    }

    with pytest.raises(ValueError, match="exact calibration IDs"):
        analyze_human_taste_agreement(
            key=key,
            humans=human,
            minimum_records=1,
            minimum_annotators=2,
            minimum_kappa=0.8,
        )
