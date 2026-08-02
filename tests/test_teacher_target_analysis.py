import pytest

from novelty_distill.data.teacher_views import (
    TeacherGeneration,
    build_teacher_target_artifact,
)
from novelty_distill.evaluation.teacher_target_analysis import (
    render_teacher_target_markdown,
    summarize_teacher_targets,
)


def test_teacher_target_summary_exposes_view_selection_and_thresholds() -> None:
    clusters = ("a", "a", "a", "b", "b", "c", "d", "e")
    qualities = (0.2, 0.9, 0.5, 0.8, 0.7, 0.6, 0.4, 0.3)
    generations = tuple(
        TeacherGeneration(
            prompt_id="p1",
            sample_index=index,
            text=f"sample-{index}",
            quality_score=qualities[index],
            cluster_id=clusters[index],
        )
        for index in range(8)
    )
    targets = build_teacher_target_artifact(generations, seed=17)
    scores = (
        {
            "prompt_id": "p1",
            "records": [
                {
                    "prompt_id": "p1",
                    "sample_index": index,
                    "text": f"sample-{index}",
                    "finish_reason": "length" if index == 7 else "stop",
                    "completion_tokens": 100 + index,
                }
                for index in range(8)
            ],
        },
    )
    metadata = {
        "prompt_diagnostics": {
            "p1": {
                "raw": {"clusters_by_threshold": {"0.700": 2, "0.820": 4}},
                "instructed": {"clusters_by_threshold": {"0.700": 3, "0.820": 5}},
            }
        }
    }

    summary = summarize_teacher_targets(
        generations=generations,
        score_payloads=scores,
        target_artifact=targets,
        cluster_metadata=metadata,
        seed=17,
    )

    assert summary["num_prompts"] == 1
    assert summary["views"]["best1"]["quality_mean"] == pytest.approx(0.9)
    assert summary["views"]["diverse4"]["quality_mean"] == pytest.approx(0.675)
    assert summary["views"]["diverse4"]["four_primary_clusters_rate"] == 1
    assert summary["views"]["all8"]["length_stop_rate"] == pytest.approx(1 / 8)
    assert summary["single_view_overlap"]["mode1_equals_best1"]["rate"] == 1
    assert summary["thresholds"]["0.700"]["instructed_cluster_mean"] == 3
    assert summary["thresholds"]["0.700"]["instructed_greater_count"] == 1
    markdown = render_teacher_target_markdown(summary)
    assert "# Frozen teacher-target findings" in markdown
    assert "not expert novelty labels" in markdown
    assert "| `diverse4` |" in markdown
