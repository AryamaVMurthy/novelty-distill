import hashlib
import json
from pathlib import Path

import pytest

from novelty_distill.evaluation.official_results import (
    OfficialEvaluationSummary,
    combine_official_summaries,
    hypospace_metrics,
    noveltybench_metrics,
    validate_official_summary,
)


def test_noveltybench_summary_requires_complete_official_metrics() -> None:
    results = {
        "total_samples": 100,
        "completed_samples": 100,
        "scores": [
            {
                "name": name,
                "scored_samples": 100,
                "unscored_samples": 0,
                "metrics": {
                    "mean": {"value": mean},
                    "stderr": {"value": 0.1},
                },
            }
            for name, mean in (("distinct_k", 5.0), ("utility_k", 3.0))
        ],
    }

    metrics = noveltybench_metrics(results, expected_samples=100)

    assert metrics["distinct_k_mean"] == 5.0
    assert metrics["utility_k_stderr"] == 0.1


def test_hypospace_summary_rejects_swallowed_errors() -> None:
    payload = {
        "n_samples": 9,
        "n_queries_per_sample": 10,
        "error_summary": {"total_errors": 1},
        "per_sample_results": [{}] * 9,
        "statistics": {},
    }

    with pytest.raises(ValueError, match="errors"):
        hypospace_metrics(payload, expected_samples=9, num_generations=10)


def test_official_summary_resume_binds_model_and_controls(tmp_path: Path) -> None:
    artifact = tmp_path / "result.eval"
    artifact.write_bytes(b"result")
    summary_path = tmp_path / "summary.json"
    summary = OfficialEvaluationSummary(
        eval_id="B1-final",
        suite="noveltybench",
        model_identity="sha256:model",
        expected_samples=100,
        num_generations=10,
        artifact=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        metrics={"distinct_k_mean": 4.0},
    )
    summary_path.write_text(json.dumps(summary.model_dump(mode="json")), encoding="utf-8")

    assert validate_official_summary(
        summary_path,
        eval_id="B1-final",
        suite="noveltybench",
        domain=None,
        model_identity="sha256:model",
        expected_samples=100,
        num_generations=10,
    )
    with pytest.raises(ValueError, match="controls changed"):
        validate_official_summary(
            summary_path,
            eval_id="B1-final",
            suite="noveltybench",
            domain=None,
            model_identity="different",
            expected_samples=100,
            num_generations=10,
        )


def test_combined_official_suite_requires_every_domain_and_one_model(tmp_path: Path) -> None:
    paths = []
    for index, (suite, domain) in enumerate(
        (
            ("noveltybench", None),
            ("hypospace", "causal"),
            ("hypospace", "3d"),
            ("hypospace", "boolean"),
        )
    ):
        artifact = tmp_path / f"artifact-{index}"
        artifact.write_text("result", encoding="utf-8")
        summary = OfficialEvaluationSummary(
            eval_id="model-final",
            suite=suite,
            domain=domain,
            model_identity="sha256:model",
            expected_samples=100 if suite == "noveltybench" else 1,
            num_generations=10,
            artifact=str(artifact),
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            metrics={"score": float(index)},
        )
        path = tmp_path / f"summary-{index}.json"
        path.write_text(json.dumps(summary.model_dump(mode="json")), encoding="utf-8")
        paths.append(path)

    combined = combine_official_summaries(paths, eval_id="model-final")

    assert combined["model_identity"] == "sha256:model"
    assert len(combined["metrics"]) == 4
