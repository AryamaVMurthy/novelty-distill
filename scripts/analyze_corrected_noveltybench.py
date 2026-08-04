#!/usr/bin/env python3
"""Build a validated prompt-paired matrix from corrected NoveltyBench logs.

Run this script in an environment containing the exact Inspect version used by
the benchmark, for example::

    uv run --with inspect-ai==0.3.233 python scripts/analyze_corrected_noveltybench.py ...
"""

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.noveltybench_matrix import (
    paired_noveltybench_contrasts,
)
from novelty_distill.evaluation.official_results import (
    OfficialEvaluationSummary,
    validate_official_summary,
)
from novelty_distill.training.provenance import atomic_json


def _binding(value: str) -> tuple[str, Path]:
    method, separator, raw_path = value.partition("=")
    if not separator or not method.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected METHOD=SUMMARY_JSON")
    return method.strip(), Path(raw_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_method(
    summary_path: Path,
) -> tuple[OfficialEvaluationSummary, dict[str, dict[str, float]]]:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as error:  # pragma: no cover - exercised by the CLI environment
        raise RuntimeError("inspect-ai==0.3.233 is required to read .eval artifacts") from error

    summary = OfficialEvaluationSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    validate_official_summary(
        summary_path,
        eval_id=summary.eval_id,
        suite="noveltybench",
        domain=None,
        model_identity=summary.model_identity,
        expected_samples=summary.expected_samples,
        num_generations=summary.num_generations,
        novelty_base_seed=(
            summary.sampling_protocol.base_seed if summary.sampling_protocol else None
        ),
    )
    artifact_path = Path(summary.artifact)
    if not artifact_path.is_absolute():
        artifact_path = summary_path.parent / artifact_path
    log = read_eval_log(artifact_path)
    if log.status != "success" or log.samples is None:
        raise ValueError(f"NoveltyBench log is not successful: {artifact_path}")
    scores: dict[str, dict[str, float]] = {}
    for sample in log.samples:
        sample_id = str(sample.id)
        scorer = (sample.scores or {}).get("novelty_bench_scorer")
        value: Any = scorer.value if scorer is not None else None
        if not isinstance(value, Mapping):
            raise ValueError(f"NoveltyBench sample {sample_id} has no scorer values")
        if sample_id in scores:
            raise ValueError(f"NoveltyBench sample ID is duplicated: {sample_id}")
        scores[sample_id] = {
            "distinct_k": float(value["distinct_k"]),
            "utility_k": float(value["utility_k"]),
        }
    if len(scores) != summary.expected_samples:
        raise ValueError(
            f"NoveltyBench sample count is {len(scores)}, expected {summary.expected_samples}"
        )
    return summary, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_binding, required=True)
    parser.add_argument("--baseline", default="A0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(args.run)
    if len(paths) != len(args.run):
        raise ValueError("method names must be unique")
    summaries: dict[str, OfficialEvaluationSummary] = {}
    scores_by_method: dict[str, dict[str, dict[str, float]]] = {}
    for method, path in paths.items():
        summary, scores = _load_method(path)
        summaries[method] = summary
        scores_by_method[method] = scores
    protocols = {
        summary.sampling_protocol.model_dump_json()
        if summary.sampling_protocol is not None
        else ""
        for summary in summaries.values()
    }
    populations = {
        (summary.expected_samples, summary.num_generations)
        for summary in summaries.values()
    }
    if len(protocols) != 1 or len(populations) != 1:
        raise ValueError("NoveltyBench runs do not share one evaluation protocol")

    payload = {
        "schema_version": 1,
        "benchmark": "yimingzhang/novelty-bench",
        "claim_boundary": (
            "Measures generic functional response diversity and benchmark utility; "
            "it does not establish scientific-idea novelty."
        ),
        "baseline": args.baseline,
        "protocol": next(iter(summaries.values())).sampling_protocol.model_dump(
            mode="json"
        ),
        "methods": {
            method: {
                "eval_id": summary.eval_id,
                "model_identity": summary.model_identity,
                "summary_path": str(paths[method]),
                "summary_sha256": _sha256(paths[method]),
                "artifact": summary.artifact,
                "artifact_sha256": summary.artifact_sha256,
                "metrics": summary.metrics,
                "sampling_diagnostics": summary.sampling_diagnostics.model_dump(
                    mode="json"
                ),
            }
            for method, summary in sorted(summaries.items())
        },
        "paired_contrasts": paired_noveltybench_contrasts(
            scores_by_method, baseline=args.baseline
        ),
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
