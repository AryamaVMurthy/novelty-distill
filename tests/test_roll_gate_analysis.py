import json
from pathlib import Path

import pytest

from novelty_distill.evaluation.roll_gate import compare_clustered_roll_runs


def _write_run(path: Path, *, modes: tuple[str, ...], quality: tuple[float, ...]) -> None:
    records = [
        {
            "prompt_id": "p1",
            "sample_index": index,
            "text": f"answer-{index}",
            "quality_score": score,
            "cluster_id": mode,
        }
        for index, (mode, score) in enumerate(zip(modes, quality, strict=True))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    dimensions = [
        {
            "relevance": 4,
            "feasibility": 4 if score >= 0.5 else 3,
            "soundness": 4,
            "clarity": 4,
            "instruction_compliance": 4,
        }
        for score in quality
    ]
    path.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps({"prompt_diagnostics": {"p1": {"judge_dimensions": dimensions}}})
    )


def test_roll_gate_compares_validity_diversity_and_control(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary.jsonl"
    seeded = tmp_path / "seeded.jsonl"
    _write_run(
        ordinary,
        modes=("a", "a", "a", "a", "a", "a", "a", "a"),
        quality=(0.8,) * 8,
    )
    _write_run(
        seeded,
        modes=("a", "a", "b", "b", "c", "c", "d", "d"),
        quality=(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4),
    )

    result = compare_clustered_roll_runs(
        ordinary_path=ordinary, seeded_path=seeded, input_seed_repeats=2
    )

    assert result["ordinary"]["semantic_clusters_mean"] == 1
    assert result["seeded"]["semantic_clusters_mean"] == 4
    assert result["seeded"]["validity_rate"] == pytest.approx(0.75)
    assert result["seeded"]["roll_mode_mutual_information_bits_mean"] == 2
    assert result["seeded"]["within_roll_mode_agreement_mean"] == 1
    assert result["deltas"]["semantic_clusters_mean"] == 3
    assert result["deltas"]["validity_rate"] == pytest.approx(-0.25)
