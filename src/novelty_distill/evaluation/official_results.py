"""Fail-closed summaries for pinned NoveltyBench and HypoSpace artifacts."""

import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class NoveltyBenchSamplingProtocol(BaseModel):
    """Sampling controls required for reproducible independent benchmark draws."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["independent-per-generation-seed-v1"] = (
        "independent-per-generation-seed-v1"
    )
    base_seed: int = Field(ge=0)
    seed_stride: Literal[1] = 1
    temperature: Literal[1.0] = 1.0
    top_p: Literal[1.0] = 1.0


class NoveltyBenchSamplingDiagnostics(BaseModel):
    """Artifact-derived evidence that the requested seed schedule was observed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duplicate_completion_prompt_count: int = Field(ge=0)
    mean_unique_completions: float = Field(ge=1)
    minimum_unique_completions: int = Field(ge=1)
    sampling_base_seed: int = Field(ge=0)
    sampling_seed_stride: Literal[1] = 1


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
    sampling_protocol: NoveltyBenchSamplingProtocol | None = None
    sampling_diagnostics: NoveltyBenchSamplingDiagnostics | None = None


def validate_official_summary(
    path: Path,
    *,
    eval_id: str,
    suite: str,
    domain: str | None,
    model_identity: str,
    expected_samples: int,
    num_generations: int,
    novelty_base_seed: int | None = None,
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
    if suite == "noveltybench":
        if summary.sampling_protocol is None or summary.sampling_diagnostics is None:
            raise ValueError("NoveltyBench summary has no independent sampling protocol")
        if (
            novelty_base_seed is not None
            and summary.sampling_protocol.base_seed != novelty_base_seed
        ):
            raise ValueError(f"official evaluation controls changed for {path}")
        if summary.sampling_diagnostics.sampling_base_seed != summary.sampling_protocol.base_seed:
            raise ValueError("NoveltyBench sampling diagnostics contradict the protocol")
    elif summary.sampling_protocol is not None or summary.sampling_diagnostics is not None:
        raise ValueError("HypoSpace summary unexpectedly contains NoveltyBench sampling fields")
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


def noveltybench_sampling_diagnostics(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_samples: int,
    num_generations: int,
    base_seed: int,
) -> dict[str, float | int]:
    """Prove that every NoveltyBench completion used its declared independent seed."""

    if expected_samples <= 0 or num_generations <= 0 or base_seed < 0:
        raise ValueError("NoveltyBench sampling controls are invalid")
    materialized = tuple(records)
    if len(materialized) != expected_samples:
        raise ValueError("NoveltyBench sampling records are incomplete")
    expected_seeds = tuple(range(base_seed, base_seed + num_generations))
    unique_counts: list[int] = []
    for record in materialized:
        sample_id = str(record.get("sample_id", "")).strip()
        completions = record.get("completions")
        declared_seeds = record.get("declared_seeds")
        observed_seeds = record.get("observed_seeds")
        if not sample_id:
            raise ValueError("NoveltyBench sampling record has no sample ID")
        if not isinstance(completions, list) or len(completions) != num_generations:
            raise ValueError(f"NoveltyBench completion count changed for {sample_id}")
        if any(
            not isinstance(completion, str) or not completion.strip()
            for completion in completions
        ):
            raise ValueError(f"NoveltyBench has an empty completion for {sample_id}")
        if tuple(declared_seeds or ()) != expected_seeds:
            raise ValueError(f"NoveltyBench declared generation seeds changed for {sample_id}")
        if tuple(observed_seeds or ()) != expected_seeds:
            raise ValueError(f"NoveltyBench observed generation seeds changed for {sample_id}")
        unique_counts.append(len(set(completions)))
    return {
        "duplicate_completion_prompt_count": sum(
            count < num_generations for count in unique_counts
        ),
        "mean_unique_completions": statistics.fmean(unique_counts),
        "minimum_unique_completions": min(unique_counts),
        "sampling_base_seed": base_seed,
        "sampling_seed_stride": 1,
    }


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
