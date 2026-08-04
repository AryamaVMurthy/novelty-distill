from pathlib import Path

import yaml

from novelty_distill.evaluation.contrasts import (
    analyze_contrasts,
    analyze_threshold_directions,
    summarize_method_metrics,
)
from novelty_distill.evaluation.evidence_policy import (
    EvidencePolicy,
    annotate_method_summaries,
)
from novelty_distill.evaluation.reporting import render_contrast_markdown

ROOT = Path(__file__).resolve().parents[1]


def test_primary_analysis_config_executes_as_one_complete_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/evaluation/primary_contrasts.yaml").read_text(encoding="utf-8")
    )
    methods = sorted(
        {
            "A0",
            "A1",
            "A3",
            *(
                str(contrast[field])
                for contrast in config["contrasts"]
                for field in ("reference", "treatment")
            ),
        }
    )
    descriptive_metrics = tuple(config["descriptive_metrics"])
    rows = tuple(
        {
            "method": method,
            "prompt_id": prompt_id,
            **{
                metric: 0.1 + method_index * 0.001 + prompt_index * 0.01
                for metric in descriptive_metrics
            },
        }
        for method_index, method in enumerate(methods)
        for prompt_index, prompt_id in enumerate(("p1", "p2", "p3"))
    )
    threshold_rows = tuple(
        {
            "method": method,
            "threshold": threshold,
            "prompt_id": prompt_id,
            **{
                metric: 0.1 + method_index * 0.001 + prompt_index * 0.01
                for metric in config["threshold_metric_directions"]
            },
        }
        for method_index, method in enumerate(methods)
        for threshold in ("0.700", "0.820")
        for prompt_index, prompt_id in enumerate(("p1", "p2", "p3"))
    )
    results = analyze_contrasts(
        rows=rows,
        contrasts=tuple(config["contrasts"]),
        metrics=tuple(config["metrics"]),
        bootstrap_samples=100,
        seed=17,
    )
    directions = analyze_threshold_directions(
        rows=threshold_rows,
        contrasts=tuple(config["contrasts"]),
        metric_directions=config["threshold_metric_directions"],
    )
    payload = {
        "schema_version": 2,
        "bootstrap_samples": 100,
        "seed": 17,
        "git_commit": "a" * 40,
        "evidence_policy": {
            "policy": "test-policy-v1",
            "sha256": "b" * 64,
            "claim_boundary": "Quarantined methods are descriptive only.",
        },
        "method_summaries": annotate_method_summaries(
            summaries=summarize_method_metrics(rows=rows, metrics=descriptive_metrics),
            policy=EvidencePolicy(
                name="test-policy-v1",
                quarantined_methods={"A3": "contaminated target"},
                claim_boundary="Quarantined methods are descriptive only.",
            ),
        ),
        "results": results,
        "threshold_direction_counts": directions,
    }

    report = render_contrast_markdown(payload)

    assert len(results) == len(config["metrics"])
    assert len(payload["method_summaries"]) == len(methods)
    assert "`student_instruction_compliance_mean` mean" in report
    assert "`A1`" in report
    assert "`A3`" in report
    assert "quarantined" in report


def test_repository_primary_contrasts_exclude_quarantined_methods() -> None:
    policy = EvidencePolicy.from_path(
        ROOT / "configs/evaluation/evidence_status.yaml"
    )
    for filename in ("primary_contrasts.yaml", "compact_baseline_contrasts.yaml"):
        config = yaml.safe_load(
            (ROOT / "configs/evaluation" / filename).read_text(encoding="utf-8")
        )
        policy.validate_primary_contrasts(tuple(config["contrasts"]))
