import json
import subprocess
import sys

import pytest

from novelty_distill.evaluation.matrix import (
    aggregate_seeded_evaluation_metrics,
    collect_prompt_metric_rows,
    collect_threshold_metric_rows,
)


def _evaluation(offset: float = 0.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "prompt_metrics": {
            "prompt-b": {"quality": 0.7 + offset, "recall": 0.5},
            "prompt-a": {"quality": 0.8 + offset, "recall": 1.0},
        },
        "prompt_metrics_by_threshold": {
            "0.700": {
                "prompt-b": {"quality": 0.7 + offset, "recall": 0.5},
                "prompt-a": {"quality": 0.8 + offset, "recall": 1.0},
            },
            "0.820": {
                "prompt-b": {"quality": 0.6 + offset, "recall": 0.25},
                "prompt-a": {"quality": 0.7 + offset, "recall": 0.75},
            },
        },
    }


def test_collect_prompt_metric_rows_aligns_methods_and_prompts() -> None:
    rows = collect_prompt_metric_rows({"A0": _evaluation(), "B3": _evaluation(offset=0.1)})

    assert rows == (
        {"method": "A0", "prompt_id": "prompt-a", "quality": 0.8, "recall": 1.0},
        {"method": "A0", "prompt_id": "prompt-b", "quality": 0.7, "recall": 0.5},
        {
            "method": "B3",
            "prompt_id": "prompt-a",
            "quality": pytest.approx(0.9),
            "recall": 1.0,
        },
        {
            "method": "B3",
            "prompt_id": "prompt-b",
            "quality": pytest.approx(0.8),
            "recall": 0.5,
        },
    )


def test_collect_prompt_metric_rows_rejects_unpaired_prompt_sets() -> None:
    candidate = _evaluation()
    candidate["prompt_metrics"].pop("prompt-b")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="prompt IDs"):
        collect_prompt_metric_rows({"A0": _evaluation(), "B3": candidate})


def test_collect_prompt_metric_rows_rejects_changed_metric_schema() -> None:
    candidate = _evaluation()
    candidate["prompt_metrics"]["prompt-a"].pop("recall")  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match="metric names"):
        collect_prompt_metric_rows({"A0": _evaluation(), "B3": candidate})


def test_collect_prompt_metric_rows_rejects_changed_teacher_partition() -> None:
    reference = _evaluation()
    candidate = _evaluation(offset=0.1)
    for payload in (reference, candidate):
        for metrics in payload["prompt_metrics"].values():  # type: ignore[union-attr]
            metrics["teacher_semantic_clusters"] = 2
    candidate["prompt_metrics"]["prompt-a"]["teacher_semantic_clusters"] = 1  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match="teacher partition"):
        collect_prompt_metric_rows({"A0": reference, "B3": candidate})


def test_collect_threshold_metric_rows_aligns_methods_thresholds_and_prompts() -> None:
    rows = collect_threshold_metric_rows({"A0": _evaluation(), "B3": _evaluation(offset=0.1)})

    assert len(rows) == 8
    assert rows[0] == {
        "method": "A0",
        "threshold": "0.700",
        "prompt_id": "prompt-a",
        "quality": 0.8,
        "recall": 1.0,
    }
    assert rows[-1] == {
        "method": "B3",
        "threshold": "0.820",
        "prompt_id": "prompt-b",
        "quality": pytest.approx(0.7),
        "recall": 0.25,
    }


def test_collect_threshold_metric_rows_rejects_incomplete_curves() -> None:
    candidate = _evaluation(offset=0.1)
    candidate["prompt_metrics_by_threshold"].pop("0.700")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="thresholds"):
        collect_threshold_metric_rows({"A0": _evaluation(), "B3": candidate})


def test_collect_threshold_metric_rows_rejects_changed_teacher_partition() -> None:
    reference = _evaluation()
    candidate = _evaluation(offset=0.1)
    for payload in (reference, candidate):
        for prompts in payload["prompt_metrics_by_threshold"].values():  # type: ignore[union-attr]
            for metrics in prompts.values():
                metrics["teacher_semantic_clusters"] = 2
    candidate["prompt_metrics_by_threshold"]["0.820"]["prompt-a"][  # type: ignore[index]
        "teacher_semantic_clusters"
    ] = 1

    with pytest.raises(ValueError, match="teacher partition"):
        collect_threshold_metric_rows({"A0": reference, "B3": candidate})


def test_seed_aggregation_requires_balanced_seeds_and_averages_each_prompt() -> None:
    result = aggregate_seeded_evaluation_metrics(
        {
            "B1": {17: _evaluation(), 29: _evaluation(offset=0.2)},
            "B3": {17: _evaluation(offset=0.1), 29: _evaluation(offset=0.3)},
        },
        expected_seeds=(17, 29),
    )

    assert result["evaluations"]["B1"]["prompt_metrics"]["prompt-a"] == {
        "quality": pytest.approx(0.9),
        "recall": 1.0,
    }
    assert result["evaluations"]["B3"]["prompt_metrics_by_threshold"]["0.820"]["prompt-b"][
        "quality"
    ] == pytest.approx(0.8)
    assert result["seed_summaries"]["B1"]["17"]["quality_mean"] == pytest.approx(0.75)

    with pytest.raises(ValueError, match="exactly the expected seeds"):
        aggregate_seeded_evaluation_metrics(
            {
                "B1": {17: _evaluation(), 29: _evaluation(offset=0.2)},
                "B3": {17: _evaluation(offset=0.1)},
            },
            expected_seeds=(17, 29),
        )


def test_seeded_collector_writes_prompt_threshold_and_seed_manifests(tmp_path) -> None:
    inputs = []
    for method, offset in (("B1", 0.0), ("B3", 0.1)):
        for seed, seed_offset in ((17, 0.0), (29, 0.2)):
            path = tmp_path / f"{method}-{seed}.json"
            path.write_text(json.dumps(_evaluation(offset + seed_offset)), encoding="utf-8")
            inputs.extend(("--input", f"{method}:{seed}={path}"))
    output = tmp_path / "prompt.jsonl"
    threshold = tmp_path / "threshold.jsonl"
    manifest = tmp_path / "manifest.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_seeded_evaluation_metrics.py",
            *inputs,
            "--seeds",
            "17,29",
            "--output",
            str(output),
            "--threshold-output",
            str(threshold),
            "--manifest",
            str(manifest),
        ],
        check=True,
    )

    prompt_rows = [json.loads(line) for line in output.read_text().splitlines()]
    result = json.loads(manifest.read_text())
    assert len(prompt_rows) == 4
    assert result["expected_seeds"] == [17, 29]
    assert result["seed_summaries"]["B3"]["29"]["quality_mean"] == pytest.approx(1.05)
