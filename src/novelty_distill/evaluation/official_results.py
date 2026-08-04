"""Fail-closed summaries for pinned NoveltyBench and HypoSpace artifacts."""

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OfficialEvaluationSummary(BaseModel):
    """Canonical result used to resume and compare one official evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    eval_id: str = Field(min_length=1)
    suite: Literal["noveltybench", "hypospace"]
    domain: Literal["causal", "3d", "boolean"] | None = None
    model_identity: str = Field(min_length=1)
    expected_samples: int = Field(gt=0)
    num_generations: int = Field(gt=0)
    artifact: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float]


def validate_official_summary(
    path: Path,
    *,
    eval_id: str,
    suite: str,
    domain: str | None,
    model_identity: str,
    expected_samples: int,
    num_generations: int,
) -> bool:
    """Return false when absent and reject any stale or incompatible result."""

    if not path.exists():
        return False
    try:
        summary = OfficialEvaluationSummary.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid official evaluation summary {path}") from error
    expected = {
        "eval_id": eval_id,
        "suite": suite,
        "domain": domain,
        "model_identity": model_identity,
        "expected_samples": expected_samples,
        "num_generations": num_generations,
    }
    actual = {key: getattr(summary, key) for key in expected}
    if actual != expected:
        raise ValueError(f"official evaluation controls changed for {path}")
    if not summary.metrics or any(not math.isfinite(value) for value in summary.metrics.values()):
        raise ValueError(f"official evaluation metrics are invalid in {path}")
    artifact = Path(summary.artifact)
    if not artifact.is_file():
        raise ValueError(f"official evaluation artifact is missing: {artifact}")
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != summary.artifact_sha256:
        raise ValueError(f"official evaluation artifact changed: {artifact}")
    return True


def noveltybench_metrics(results: Mapping[str, Any], *, expected_samples: int) -> dict[str, float]:
    """Extract the two official NoveltyBench aggregates after completeness checks."""

    if (
        results.get("total_samples") != expected_samples
        or results.get("completed_samples") != expected_samples
    ):
        raise ValueError("NoveltyBench did not complete the expected sample population")
    scores = results.get("scores")
    if not isinstance(scores, list):
        raise ValueError("NoveltyBench result has no score list")
    metrics: dict[str, float] = {}
    for score in scores:
        if not isinstance(score, Mapping) or score.get("name") not in {
            "distinct_k",
            "utility_k",
        }:
            continue
        if score.get("scored_samples") != expected_samples or score.get("unscored_samples") != 0:
            raise ValueError("NoveltyBench contains unscored samples")
        aggregates = score.get("metrics")
        if not isinstance(aggregates, Mapping):
            raise ValueError("NoveltyBench score has no aggregate metrics")
        name = str(score["name"])
        for statistic in ("mean", "stderr"):
            entry = aggregates.get(statistic)
            if not isinstance(entry, Mapping):
                raise ValueError(f"NoveltyBench {name} has no {statistic}")
            value = float(entry["value"])
            if not math.isfinite(value):
                raise ValueError(f"NoveltyBench {name} {statistic} is not finite")
            metrics[f"{name}_{statistic}"] = value
    if set(metrics) != {
        "distinct_k_mean",
        "distinct_k_stderr",
        "utility_k_mean",
        "utility_k_stderr",
    }:
        raise ValueError("NoveltyBench is missing required official metrics")
    return metrics


def hypospace_metrics(
    payload: Mapping[str, Any], *, expected_samples: int, num_generations: int
) -> dict[str, float]:
    """Extract official HypoSpace rates and reject swallowed provider failures."""

    if payload.get("n_samples") != expected_samples:
        raise ValueError("HypoSpace sample count changed")
    if payload.get("n_queries_per_sample") != num_generations:
        raise ValueError("HypoSpace query count changed")
    error_summary = payload.get("error_summary")
    if not isinstance(error_summary, Mapping) or error_summary.get("total_errors") != 0:
        raise ValueError("HypoSpace reported provider or parsing errors")
    per_sample = payload.get("per_sample_results")
    if not isinstance(per_sample, list) or len(per_sample) != expected_samples:
        raise ValueError("HypoSpace per-sample results are incomplete")
    statistics = payload.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("HypoSpace result has no statistics")
    fields = {
        "parse_success_rate": "parse_success_rate",
        "valid_rate": "validity_rate",
        "novelty_rate": "uniqueness_rate",
        "recovery_rate": "recovery_rate",
    }
    metrics: dict[str, float] = {}
    for source, destination in fields.items():
        entry = statistics.get(source)
        if source == "parse_success_rate" and entry is None:
            # The pinned Boolean CLI has no parser-stage statistic. Its
            # zero-error summary plus complete per-sample records means every
            # request completed; expose that contract explicitly so Boolean
            # remains comparable in the combined four-component report.
            entry = {"mean": 1.0}
        if not isinstance(entry, Mapping):
            raise ValueError(f"HypoSpace has no {source}")
        value = float(entry["mean"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"HypoSpace {source} is outside [0, 1]")
        metrics[destination] = value
    return metrics


def combine_official_summaries(
    paths: list[Path] | tuple[Path, ...], *, eval_id: str
) -> dict[str, Any]:
    """Combine the complete NoveltyBench plus three-domain HypoSpace suite."""

    summaries = tuple(
        OfficialEvaluationSummary.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    )
    for path, summary in zip(paths, summaries, strict=True):
        validate_official_summary(
            path,
            eval_id=summary.eval_id,
            suite=summary.suite,
            domain=summary.domain,
            model_identity=summary.model_identity,
            expected_samples=summary.expected_samples,
            num_generations=summary.num_generations,
        )
    expected_keys = {
        ("noveltybench", None),
        ("hypospace", "causal"),
        ("hypospace", "3d"),
        ("hypospace", "boolean"),
    }
    keys = {(summary.suite, summary.domain) for summary in summaries}
    if len(summaries) != 4 or keys != expected_keys:
        raise ValueError("official suite requires NoveltyBench and all three HypoSpace domains")
    if {summary.eval_id for summary in summaries} != {eval_id}:
        raise ValueError("official suite evaluation IDs differ")
    identities = {summary.model_identity for summary in summaries}
    if len(identities) != 1:
        raise ValueError("official suite model identities differ")
    combined_metrics: dict[str, float] = {}
    inputs: dict[str, dict[str, str]] = {}
    for path, summary in zip(paths, summaries, strict=True):
        namespace = summary.suite if summary.domain is None else f"{summary.suite}_{summary.domain}"
        for name, value in summary.metrics.items():
            combined_metrics[f"{namespace}.{name}"] = value
        inputs[namespace] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": 1,
        "eval_id": eval_id,
        "model_identity": identities.pop(),
        "metrics": dict(sorted(combined_metrics.items())),
        "inputs": dict(sorted(inputs.items())),
    }
