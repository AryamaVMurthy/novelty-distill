from novelty_distill.evaluation.reporting import render_contrast_markdown


def test_contrast_report_renders_inference_and_threshold_direction_counts() -> None:
    payload = {
        "schema_version": 2,
        "bootstrap_samples": 10_000,
        "seed": 17,
        "results": {
            "recall": {
                "B-vs-A": {
                    "n": 3,
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
    assert "| `B-vs-A` | 3 | 0.1 | [0.01, 0.2] | 0.5 | 0.02 | 0.04 |" in report
    assert "0.700: 2/0/1" in report
    assert "not human-validated scientific novelty labels" in report
