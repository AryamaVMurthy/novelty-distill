import hashlib
import json
from pathlib import Path

import pytest

from novelty_distill.training.metadata_migration import backfill_declared_divergence


def _registry(path: Path) -> None:
    path.write_text(
        """\
schema_version: 1
baselines:
  - id: C1-best1
    name: forward-kl
    family: off_policy
    backend: trl_gkd
    official_source: huggingface/trl
    trajectory_source: teacher
    target_view: best1
    divergence: forward_kl
    lmbda: 0.0
    beta: 0.0
""",
        encoding="utf-8",
    )


def _metadata(path: Path, *, beta: float = 0.0) -> bytes:
    content = json.dumps(
        {
            "baseline_id": "C1-best1",
            "backend": "trl_gkd",
            "target_view": "best1",
            "lmbda": 0.0,
            "beta": beta,
        },
        indent=1,
    ).encode()
    path.write_bytes(content)
    return content


def test_divergence_backfill_preserves_exact_original_and_is_idempotent(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "baselines.yaml"
    _registry(registry)
    metadata = tmp_path / "run_metadata.json"
    original = _metadata(metadata)

    result = backfill_declared_divergence(
        registry, {"C1-best1": metadata}, migration_commit="a" * 40
    )

    backup = tmp_path / "run_metadata.pre-divergence-backfill.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert result["records"]["C1-best1"]["status"] == "migrated"
    assert backup.read_bytes() == original
    assert payload["divergence"] == "forward_kl"
    assert payload["metadata_migrations"] == [
        {
            "schema_version": 1,
            "kind": "backfill_declared_divergence",
            "field": "divergence",
            "value": "forward_kl",
            "evidence": {"lmbda": 0.0, "beta": 0.0},
            "prior_sha256": hashlib.sha256(original).hexdigest(),
            "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
            "migration_commit": "a" * 40,
        }
    ]
    migrated = metadata.read_bytes()

    repeated = backfill_declared_divergence(
        registry, {"C1-best1": metadata}, migration_commit="a" * 40
    )
    assert repeated["records"]["C1-best1"]["status"] == "already_migrated"
    assert metadata.read_bytes() == migrated
    assert backup.read_bytes() == original


def test_divergence_backfill_rejects_ambiguous_or_conflicting_evidence(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "baselines.yaml"
    _registry(registry)
    metadata = tmp_path / "run_metadata.json"
    _metadata(metadata, beta=1.0)

    with pytest.raises(ValueError, match="legacy beta"):
        backfill_declared_divergence(
            registry, {"C1-best1": metadata}, migration_commit="a" * 40
        )


def test_divergence_backfill_accepts_current_schema_without_mutation(tmp_path: Path) -> None:
    registry = tmp_path / "baselines.yaml"
    _registry(registry)
    metadata = tmp_path / "run_metadata.json"
    _metadata(metadata)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["divergence"] = "forward_kl"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    original = metadata.read_bytes()

    result = backfill_declared_divergence(
        registry, {"C1-best1": metadata}, migration_commit="a" * 40
    )

    assert result["records"]["C1-best1"]["status"] == "not_required"
    assert metadata.read_bytes() == original
    assert not (tmp_path / "run_metadata.pre-divergence-backfill.json").exists()

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["beta"] = 0.0
    payload["divergence"] = "reverse_kl"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="existing divergence disagrees"):
        backfill_declared_divergence(
            registry, {"C1-best1": metadata}, migration_commit="a" * 40
        )
