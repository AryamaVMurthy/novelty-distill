from novelty_distill.evaluation.reporting import render_contrast_markdown


def test_contrast_report_renders_inference_and_threshold_direction_counts() -> None:
    payload = {
        "schema_version": 2,
        "bootstrap_samples": 10_000,
        "seed": 17,
        "git_commit": "a" * 40,
        "evidence_policy": {
            "policy": "posthoc-validity-quarantine-v1",
            "sha256": "b" * 64,
            "claim_boundary": "Quarantined methods are descriptive only.",
        },
        "method_summaries": {
            "A": {
                "n": 3,
                "recall_mean": 0.4,
                "recall_median": 0.4,
                "evidence_status": "primary_eligible",
            },
            "B": {
                "n": 3,
                "recall_mean": 0.5,
                "recall_median": 0.5,
                "evidence_status": "quarantined",
                "evidence_note": "target contamination",
            },
        },
        "results": {
            "recall": {
                "B-vs-A": {
                    "n": 3,
                    "reference_mean": 0.4,
                    "treatment_mean": 0.5,
                    "mean_difference": 0.1,
                    "ci_low": 0.01,
                    "ci_high": 0.2,
                    "effect_size": 0.5,
                    "p_value": 0.02,
                    "holm_p_value": 0.04,
                }
            }
        },
        "threshold_direction_counts": {
            "recall": {
                "B-vs-A": {
                    "desirable_direction": "higher",
                    "stable_mean_direction": "favorable",
                    "thresholds": {
                        "0.700": {
                            "favorable_count": 2,
                            "tied_count": 0,
                            "unfavorable_count": 1,
                        }
                    },
                }
            }
        },
    }

    report = render_contrast_markdown(payload)

    assert "# Frozen TOMATO contrast findings" in report
    assert "Producer Git commit: `aaaaaaaa" in report
    assert "## Descriptive method levels" in report
    assert "posthoc-validity-quarantine-v1" in report
    assert "Quarantined methods are descriptive only." in report
    assert "A3 is the single historical author response (K=1)" in report
    assert "| `A` | primary_eligible | 3 | 0.4 |" in report
    assert "| `B` | quarantined | 3 | 0.5 |" in report
    assert (
        "| `B-vs-A` | 3 | 0.4 | 0.5 | 0.1 | [0.01, 0.2] | 0.5 | 0.02 | 0.04 |"
        in report
    )
    assert "0.700: 2/0/1" in report
    assert "not human-validated scientific novelty labels" in report
    assert "finite-sample plug-in estimate" in report
    assert "pointwise, not simultaneous" in report
