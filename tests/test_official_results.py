import hashlib
import json
from pathlib import Path

import pytest

from novelty_distill.evaluation.official_results import (
    NoveltyBenchSamplingDiagnostics,
    NoveltyBenchSamplingProtocol,
    OfficialEvaluationSummary,
    combine_official_summaries,
    hypospace_metrics,
    noveltybench_metrics,
    noveltybench_sampling_diagnostics,
    validate_official_summary,
)


def _sampling_fields(base_seed: int = 17) -> dict[str, object]:
    return {
        "sampling_protocol": NoveltyBenchSamplingProtocol(base_seed=base_seed),
        "sampling_diagnostics": NoveltyBenchSamplingDiagnostics(
            duplicate_completion_prompt_count=0,
            mean_unique_completions=3.0,
            minimum_unique_completions=3,
            sampling_base_seed=base_seed,
        ),
    }


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


def test_noveltybench_sampling_requires_independent_observed_request_seeds() -> None:
    records = [
        {
            "sample_id": "curated-0",
            "completions": ["first", "second", "third"],
            "declared_seeds": [17, 18, 19],
            "observed_seeds": [17, 18, 19],
        }
    ]

    diagnostics = noveltybench_sampling_diagnostics(
        records,
        expected_samples=1,
        num_generations=3,
        base_seed=17,
    )

    assert diagnostics == {
        "duplicate_completion_prompt_count": 0,
        "mean_unique_completions": 3.0,
        "minimum_unique_completions": 3,
        "sampling_base_seed": 17,
        "sampling_seed_stride": 1,
    }

    records[0]["observed_seeds"] = [17, 17, 17]
    with pytest.raises(ValueError, match="observed generation seeds"):
        noveltybench_sampling_diagnostics(
            records,
            expected_samples=1,
            num_generations=3,
            base_seed=17,
        )


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


def test_hypospace_boolean_schema_uses_explicit_parser_completion_fallback() -> None:
    payload = {
        "n_samples": 35,
        "n_queries_per_sample": 10,
        "error_summary": {"total_errors": 0},
        "per_sample_results": [{}] * 35,
        "statistics": {
            "valid_rate": {"mean": 0.64},
            "novelty_rate": {"mean": 0.13},
            "recovery_rate": {"mean": 0.25},
        },
    }

    metrics = hypospace_metrics(payload, expected_samples=35, num_generations=10)

    assert metrics["parse_success_rate"] == 1.0
    assert metrics["validity_rate"] == 0.64


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
        **_sampling_fields(),
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


def test_official_summary_resolves_artifact_relative_to_summary(tmp_path: Path) -> None:
    artifact = tmp_path / "result.eval"
    artifact.write_bytes(b"portable-result")
    summary_path = tmp_path / "summary.json"
    summary = OfficialEvaluationSummary(
        eval_id="portable",
        suite="noveltybench",
        model_identity="sha256:model",
        expected_samples=1,
        num_generations=3,
        artifact="result.eval",
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        metrics={"distinct_k_mean": 2.0},
        **_sampling_fields(),
    )
    summary_path.write_text(json.dumps(summary.model_dump(mode="json")), encoding="utf-8")

    assert validate_official_summary(
        summary_path,
        eval_id="portable",
        suite="noveltybench",
        domain=None,
        model_identity="sha256:model",
        expected_samples=1,
        num_generations=3,
        novelty_base_seed=17,
    )


def test_noveltybench_summary_rejects_legacy_shared_seed_protocol(tmp_path: Path) -> None:
    artifact = tmp_path / "result.eval"
    artifact.write_bytes(b"result")
    summary_path = tmp_path / "summary.json"
    summary = OfficialEvaluationSummary(
        eval_id="A0-corrected",
        suite="noveltybench",
        model_identity="sha256:model",
        expected_samples=1,
        num_generations=3,
        artifact=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        metrics={"distinct_k_mean": 2.0},
    )
    summary_path.write_text(json.dumps(summary.model_dump(mode="json")), encoding="utf-8")

    with pytest.raises(ValueError, match="independent sampling protocol"):
        validate_official_summary(
            summary_path,
            eval_id="A0-corrected",
            suite="noveltybench",
            domain=None,
            model_identity="sha256:model",
            expected_samples=1,
            num_generations=3,
            novelty_base_seed=17,
        )

    corrected = summary.model_copy(
        update=_sampling_fields()
    )
    summary_path.write_text(json.dumps(corrected.model_dump(mode="json")), encoding="utf-8")

    assert validate_official_summary(
        summary_path,
        eval_id="A0-corrected",
        suite="noveltybench",
        domain=None,
        model_identity="sha256:model",
        expected_samples=1,
        num_generations=3,
        novelty_base_seed=17,
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
            **(_sampling_fields() if suite == "noveltybench" else {}),
        )
        path = tmp_path / f"summary-{index}.json"
        path.write_text(json.dumps(summary.model_dump(mode="json")), encoding="utf-8")
        paths.append(path)

    combined = combine_official_summaries(paths, eval_id="model-final")

    assert combined["model_identity"] == "sha256:model"
    assert len(combined["metrics"]) == 4
