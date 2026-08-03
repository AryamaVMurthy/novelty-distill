import pytest

from novelty_distill.data.teacher_views import (
    TeacherGeneration,
    build_teacher_target_artifact,
    derive_random_k_texts,
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
        "clustering_linkage": "complete",
        "cosine_threshold": 0.94,
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
    assert summary["clustering_linkage"] == "complete"
    assert summary["primary_cosine_threshold"] == 0.94
    assert summary["views"]["best1"]["quality_mean"] == pytest.approx(0.9)
    assert summary["views"]["diverse4"]["quality_mean"] == pytest.approx(0.675)
    assert summary["views"]["diverse4"]["at_least_four_primary_clusters_rate"] == 1
    assert "four_primary_clusters_rate" not in summary["views"]["all8"]
    assert summary["views"]["all8"]["length_stop_rate"] == pytest.approx(1 / 8)
    assert summary["single_view_overlap"]["mode1_equals_best1"]["rate"] == 1
    assert summary["thresholds"]["0.700"]["instructed_cluster_mean"] == 3
    assert summary["thresholds"]["0.700"]["instructed_greater_count"] == 1
    markdown = render_teacher_target_markdown(summary)
    assert "# Frozen teacher-target findings" in markdown
    assert "not expert novelty labels" in markdown
    assert "| `diverse4` |" in markdown
    assert "At-least-four-cluster prompts" in markdown
    assert "complete linkage at cosine 0.94" in markdown


def test_teacher_target_summary_can_add_nested_secondary_random4() -> None:
    generations = tuple(
        TeacherGeneration(
            prompt_id="p1",
            sample_index=index,
            text=f"sample-{index}",
            quality_score=float(index) / 10,
            cluster_id=str(index),
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
                    "text": f"sample-{index}",
                    "finish_reason": "stop",
                    "completion_tokens": 100,
                }
                for index in range(8)
            ],
        },
    )
    metadata = {
        "clustering_linkage": "complete",
        "cosine_threshold": 0.94,
        "prompt_diagnostics": {
            "p1": {
                "raw": {"clusters_by_threshold": {"0.940": 8}},
                "instructed": {"clusters_by_threshold": {"0.940": 8}},
            }
        },
    }

    summary = summarize_teacher_targets(
        generations=generations,
        score_payloads=scores,
        target_artifact=targets,
        cluster_metadata=metadata,
        seed=17,
        secondary_random_k=4,
    )

    assert summary["views"]["random4"]["num_responses"] == 4
    assert summary["views"]["random4"]["primary_clusters_per_prompt_mean"] == 4
    selected = derive_random_k_texts(
        targets["targets"]["p1"]["all8"], prompt_id="p1", seed=17, k=4
    )
    assert targets["targets"]["p1"]["random1"][0] in selected
    assert summary["views"]["random4"]["quality_mean"] == pytest.approx(
        sum(int(text.removeprefix("sample-")) / 10 for text in selected) / 4
    )

    with pytest.raises(ValueError, match="between two and eight"):
        summarize_teacher_targets(
            generations=generations,
            score_payloads=scores,
            target_artifact=targets,
            cluster_metadata=metadata,
            seed=17,
            secondary_random_k=1,
        )
