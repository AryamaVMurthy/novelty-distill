import json
from pathlib import Path

from novelty_distill.evaluation.compact_reporting import write_compact_artifact_bundle


def test_compact_artifact_bundle_writes_table_and_two_svg_plots(tmp_path: Path) -> None:
    payload = {
        "method_summaries": {
            method: {
                "n": 10,
                "student_feasibility_mean_mean": 3.0 + index / 10,
                "student_soundness_mean_mean": 3.5 + index / 10,
                "teacher_mode_recall_mean": 0.4 + index / 20,
                "viable_semantic_yield_mean": 1.0 + index / 5,
            }
            for index, method in enumerate(("A0", "B1"))
        }
    }
    config = {
        "method_order": ["A0", "B1"],
        "table_metrics": [
            "student_feasibility_mean",
            "student_soundness_mean",
            "teacher_mode_recall",
            "viable_semantic_yield",
        ],
        "plots": [
            {
                "filename": "quality.svg",
                "title": "Quality",
                "panels": [
                    {"metric": "student_feasibility_mean", "label": "Feasibility", "maximum": 5},
                    {"metric": "student_soundness_mean", "label": "Soundness", "maximum": 5},
                ],
            },
            {
                "filename": "breadth.svg",
                "title": "Breadth",
                "panels": [
                    {"metric": "teacher_mode_recall", "label": "Mode recall", "maximum": 1},
                    {"metric": "viable_semantic_yield", "label": "Viable yield", "maximum": 4},
                ],
            },
        ],
    }

    manifest = write_compact_artifact_bundle(payload=payload, config=config, output_dir=tmp_path)

    assert manifest["files"] == ["baseline-summary.csv", "quality.svg", "breadth.svg"]
    assert (tmp_path / "baseline-summary.csv").read_text().splitlines()[0].startswith("method,n,")
    assert "A0,10,3" in (tmp_path / "baseline-summary.csv").read_text()
    assert "<svg" in (tmp_path / "quality.svg").read_text()
    assert "Mode recall" in (tmp_path / "breadth.svg").read_text()
    assert json.loads((tmp_path / "artifact-bundle.json").read_text()) == manifest
