#!/usr/bin/env python3
"""Run the frozen prompt-paired contrast family over collected model metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from novelty_distill.evaluation.compact_reporting import write_compact_artifact_bundle
from novelty_distill.evaluation.contrasts import (
    analyze_contrasts,
    analyze_threshold_directions,
    partition_available_contrasts,
    summarize_method_metrics,
)
from novelty_distill.evaluation.evidence_policy import (
    EvidencePolicy,
    annotate_method_summaries,
)
from novelty_distill.evaluation.reporting import render_contrast_markdown
from novelty_distill.provenance import repository_commit

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold-input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument(
        "--evidence-policy",
        type=Path,
        default=ROOT / "configs/evaluation/evidence_status.yaml",
    )
    parser.add_argument("--filter-absent-contrasts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    git_commit = repository_commit(Path(__file__).resolve().parents[1])
    rows = tuple(
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    threshold_rows = tuple(
        json.loads(line)
        for line in args.threshold_input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported contrast configuration schema")
    contrasts = tuple(config["contrasts"])
    omitted_contrasts: tuple[str, ...] = ()
    if args.filter_absent_contrasts:
        contrasts, omitted_contrasts = partition_available_contrasts(
            contrasts=contrasts,
            methods=tuple(sorted({str(row.get("method", "")) for row in rows})),
        )
    policy_bytes = args.evidence_policy.read_bytes()
    policy = EvidencePolicy.from_path(args.evidence_policy)
    policy.validate_primary_contrasts(contrasts)
    results = analyze_contrasts(
        rows=rows,
        contrasts=contrasts,
        metrics=tuple(config["metrics"]),
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    threshold_directions = analyze_threshold_directions(
        rows=threshold_rows,
        contrasts=contrasts,
        metric_directions=dict(config["threshold_metric_directions"]),
    )
    payload = {
        "schema_version": 2,
        "git_commit": git_commit,
        "study": config.get("study"),
        "claim_scope": config.get("claim_scope"),
        "sampling_note": config.get("sampling_note"),
        "bootstrap_samples": config["bootstrap_samples"],
        "seed": config["seed"],
        "holm_family": (
            "all supported preregistered contrasts within each metric"
            if args.filter_absent_contrasts
            else "all declared contrasts within each metric"
        ),
        "contrast_subset": {
            "filtered_for_promoted_methods": args.filter_absent_contrasts,
            "analyzed_ids": [str(contrast["id"]) for contrast in contrasts],
            "omitted_ids": list(omitted_contrasts),
        },
        "evidence_policy": {
            "policy": policy.name,
            "sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "claim_boundary": policy.claim_boundary,
        },
        "method_summaries": annotate_method_summaries(
            summaries=summarize_method_metrics(
                rows=rows,
                metrics=tuple(config.get("descriptive_metrics", config["metrics"])),
            ),
            policy=policy,
        ),
        "results": results,
        "threshold_direction_counts": threshold_directions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_contrast_markdown(payload), encoding="utf-8")
    artifact_manifest = None
    if artifact_config := config.get("artifact_bundle"):
        artifact_manifest = write_compact_artifact_bundle(
            payload=payload,
            config=artifact_config,
            output_dir=args.output.parent,
        )
    print(
        json.dumps(
            {
                "metrics": len(results),
                "threshold_metrics": len(threshold_directions),
                "output": str(args.output),
                "markdown": str(args.markdown),
                "artifact_bundle": artifact_manifest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
