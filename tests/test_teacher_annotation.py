import pytest

from novelty_distill.evaluation.teacher_annotation import (
    JudgeSpec,
    build_quality_judge_payload,
    cluster_cosine_embeddings,
    cosine_embedding_diagnostics,
    parse_quality_judge_response,
)


def _judge_spec() -> JudgeSpec:
    return JudgeSpec(
        model="Qwen/Qwen3-32B-FP8",
        revision="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
    )


def test_quality_judge_uses_strict_structured_output() -> None:
    payload = build_quality_judge_payload(
        prompt="Propose a testable scientific hypothesis.",
        response="A causal intervention could test the proposed mechanism.",
        spec=_judge_spec(),
    )

    assert payload["model"] == "Qwen/Qwen3-32B-FP8"
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    rubric = payload["messages"][0]["content"]
    assert "Reserve 5" in rubric
    assert "3 means" in rubric
    assert "1 means" in rubric
    assert "Score dimensions independently" in rubric
    schema = payload["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert set(schema["schema"]["required"]) == {
        "relevance",
        "feasibility",
        "soundness",
        "clarity",
        "instruction_compliance",
    }


def test_quality_judge_response_is_normalized_to_unit_interval() -> None:
    response = {
        "id": "chatcmpl-judge-1",
        "model": "Qwen/Qwen3-32B-FP8",
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": (
                        '{"relevance":5,"feasibility":4,"soundness":3,'
                        '"clarity":2,"instruction_compliance":1}'
                    )
                },
                "finish_reason": "stop",
            }
        ],
    }

    score = parse_quality_judge_response(response, _judge_spec())

    assert score.quality_score == pytest.approx(0.5)
    assert score.request_id == "chatcmpl-judge-1"


def test_cosine_threshold_clustering_is_deterministic() -> None:
    embeddings = (
        (1.0, 0.0),
        (0.99, 0.1),
        (0.0, 1.0),
        (-1.0, 0.0),
    )

    labels = cluster_cosine_embeddings(embeddings, threshold=0.9)

    assert labels == ("cluster-000", "cluster-000", "cluster-001", "cluster-002")


def test_complete_linkage_prevents_similarity_bridge_chaining() -> None:
    embeddings = (
        (1.0, 0.0),
        (0.9, 0.435889894),
        (0.6, 0.8),
    )

    labels = cluster_cosine_embeddings(embeddings, threshold=0.8)

    assert labels == ("cluster-000", "cluster-000", "cluster-001")


def test_cosine_clustering_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        cluster_cosine_embeddings(((0.0, 0.0),), threshold=0.9)


def test_cosine_diagnostics_report_similarity_and_threshold_sensitivity() -> None:
    embeddings = (
        (1.0, 0.0),
        (0.8, 0.6),
        (0.0, 1.0),
    )

    diagnostics = cosine_embedding_diagnostics(embeddings, thresholds=(0.5, 0.9))

    assert diagnostics["pair_count"] == 3
    assert diagnostics["cosine_min"] == pytest.approx(0.0)
    assert diagnostics["cosine_mean"] == pytest.approx((0.8 + 0.0 + 0.6) / 3)
    assert diagnostics["cosine_max"] == pytest.approx(0.8)
    assert diagnostics["clusters_by_threshold"] == {"0.500": 2, "0.900": 3}
