import hashlib
import json

import pytest

from novelty_distill.evaluation.audit_summary import summarize_corrected_evaluations


def _payload(*, value: float, student_hash: str = "a" * 64) -> dict[str, object]:
    threshold_metrics = {
        "quality_qualified_semantic_yield": value,
        "teacher_mode_recall": value / 10,
    }
    return {
        "schema_version": 1,
        "git_commit": "b" * 40,
        "num_prompts": 3,
        "teacher_samples_per_prompt": 4,
        "student_samples_per_prompt": 4,
        "primary_cosine_threshold": 0.94,
        "overall": threshold_metrics | {"student_feasibility_mean": 4.0},
        "threshold_sensitivity": {
            "0.900": threshold_metrics,
            "0.940": threshold_metrics,
        },
        "inputs": {
            "teacher_score_sha256": "c" * 64,
            "student_score_sha256": student_hash,
        },
        "embedding": {"model": "embedding-model", "revision": "d" * 40},
        "quality_qualified_semantic_yield": {
            "status": "corrected_secondary_descriptive",
            "quality_gate": {
                "relevance_minimum": 4,
                "feasibility_minimum": 4,
                "soundness_minimum": 4,
                "clarity_minimum": 4,
            },
        },
    }


def test_corrected_evaluation_summary_retains_full_threshold_curves() -> None:
    raw = json.dumps(_payload(value=1.25), sort_keys=True).encode()
    result = summarize_corrected_evaluations(
        payloads={"A0": json.loads(raw)},
        input_sha256={"A0": hashlib.sha256(raw).hexdigest()},
        jobs={"A0": "100"},
    )

    assert result["metric_status"]["quality_qualified_semantic_yield"] == (
        "corrected_secondary_descriptive"
    )
    assert result["single_threshold_inference_allowed"] is False
    assert result["methods"]["A0"]["overall"]["quality_qualified_semantic_yield"] == 1.25
    assert set(result["methods"]["A0"]["threshold_sensitivity"]) == {"0.900", "0.940"}
    assert result["methods"]["A0"]["job_id"] == "100"
    assert result["methods"]["A0"]["input_sha256"] == hashlib.sha256(raw).hexdigest()


def test_corrected_evaluation_summary_rejects_incompatible_methods() -> None:
    first = _payload(value=1.0)
    second = _payload(value=2.0, student_hash="e" * 64)
    second["num_prompts"] = 4

    with pytest.raises(ValueError, match="shared evaluation controls"):
        summarize_corrected_evaluations(
            payloads={"A0": first, "C1": second},
            input_sha256={"A0": "f" * 64, "C1": "0" * 64},
            jobs={"A0": "100", "C1": "101"},
        )


def test_corrected_evaluation_summary_requires_feasibility_in_gate() -> None:
    payload = _payload(value=1.0)
    del payload["quality_qualified_semantic_yield"]["quality_gate"][
        "feasibility_minimum"
    ]

    with pytest.raises(ValueError, match="feasibility-inclusive"):
        summarize_corrected_evaluations(
            payloads={"A0": payload},
            input_sha256={"A0": "f" * 64},
            jobs={"A0": "100"},
        )
