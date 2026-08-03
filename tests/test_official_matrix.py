import json
from pathlib import Path

import pytest

from novelty_distill.evaluation.official_matrix import aggregate_official_matrix


def _write(path: Path, eval_id: str, value: float) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "eval_id": eval_id,
                "model_identity": f"sha256:{eval_id}",
                "metrics": {
                    "noveltybench.distinct_k": value,
                    "hypospace_causal.validity": 1.0 - value,
                },
                "inputs": {},
            }
        ),
        encoding="utf-8",
    )


def test_official_matrix_aggregates_only_a_complete_method_seed_grid(tmp_path: Path) -> None:
    inputs: dict[tuple[str, int], Path] = {}
    for method, base in (("B1", 0.2), ("D1", 0.6)):
        for seed, offset in ((17, 0.0), (29, 0.2)):
            path = tmp_path / f"{method}-{seed}.json"
            _write(path, f"{method}-seed{seed}", base + offset)
            inputs[(method, seed)] = path

    result = aggregate_official_matrix(
        inputs,
        expected_methods=("B1", "D1"),
        expected_seeds=(17, 29),
    )

    assert result["status"] == "complete"
    assert result["method_summaries"]["B1"]["noveltybench.distinct_k"]["mean"] == pytest.approx(
        0.3
    )
    assert result["method_summaries"]["D1"]["hypospace_causal.validity"][
        "mean"
    ] == pytest.approx(0.3)
    assert len(result["runs"]) == 4

    with pytest.raises(ValueError, match="complete method-by-seed grid"):
        aggregate_official_matrix(
            {key: value for key, value in inputs.items() if key != ("D1", 29)},
            expected_methods=("B1", "D1"),
            expected_seeds=(17, 29),
        )
