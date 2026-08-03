"""Auditable, idempotent migrations for historical training metadata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from novelty_distill.config import load_baseline_registry
from novelty_distill.training.provenance import atomic_json, file_provenance


def _load_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"training metadata is not an object: {path}")
    return raw, payload


def _preserve_original(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing metadata backup differs from original bytes: {path}")
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def backfill_declared_divergence(
    registry_path: Path,
    metadata_paths: Mapping[str, Path],
    *,
    migration_commit: str,
) -> dict[str, Any]:
    """Backfill pre-schema TRL-GKD divergence while preserving original bytes."""

    if not migration_commit.strip():
        raise ValueError("metadata migration requires a repository commit")
    registry = load_baseline_registry(registry_path)
    baselines = {baseline.id: baseline for baseline in registry.baselines}
    registry_identity = file_provenance(registry_path)
    records: dict[str, Any] = {}

    for baseline_id, raw_path in sorted(metadata_paths.items()):
        baseline = baselines.get(baseline_id)
        if baseline is None:
            raise ValueError(f"metadata migration baseline is not declared: {baseline_id}")
        path = raw_path.resolve(strict=True)
        original, payload = _load_object(path)
        if payload.get("baseline_id") != baseline_id:
            raise ValueError(f"metadata baseline ID disagrees for {baseline_id}: {path}")
        if baseline.backend != "trl_gkd" or payload.get("backend") != "trl_gkd":
            raise ValueError(f"divergence backfill is restricted to legacy TRL-GKD: {path}")
        for field in ("lmbda", "beta", "target_view"):
            if payload.get(field) != getattr(baseline, field):
                raise ValueError(
                    f"legacy {field} does not prove declared divergence for {baseline_id}: {path}"
                )

        recorded = payload.get("divergence")
        backup = path.with_name("run_metadata.pre-divergence-backfill.json")
        if recorded is not None:
            if recorded != baseline.divergence:
                raise ValueError(f"existing divergence disagrees for {baseline_id}: {path}")
            migrations = payload.get("metadata_migrations")
            if not isinstance(migrations, list) or not any(
                isinstance(item, dict)
                and item.get("kind") == "backfill_declared_divergence"
                for item in migrations
            ):
                raise ValueError(
                    f"explicit divergence has no migration evidence for {baseline_id}: {path}"
                )
            if not backup.is_file():
                raise ValueError(f"migrated metadata has no preserved original: {path}")
            records[baseline_id] = {
                "status": "already_migrated",
                "metadata": str(path),
                "backup": str(backup),
            }
            continue

        prior_sha256 = hashlib.sha256(original).hexdigest()
        _preserve_original(backup, original)
        migrations = payload.get("metadata_migrations", [])
        if not isinstance(migrations, list):
            raise ValueError(f"metadata_migrations is not a list: {path}")
        payload["divergence"] = baseline.divergence
        payload["metadata_migrations"] = [
            *migrations,
            {
                "schema_version": 1,
                "kind": "backfill_declared_divergence",
                "field": "divergence",
                "value": baseline.divergence,
                "evidence": {"lmbda": baseline.lmbda, "beta": baseline.beta},
                "prior_sha256": prior_sha256,
                "registry_sha256": registry_identity["sha256"],
                "migration_commit": migration_commit,
            },
        ]
        atomic_json(path, payload)
        records[baseline_id] = {
            "status": "migrated",
            "metadata": str(path),
            "backup": str(backup),
            "prior_sha256": prior_sha256,
            "current_sha256": file_provenance(path)["sha256"],
        }

    return {
        "schema_version": 1,
        "status": "complete",
        "migration": "backfill_declared_divergence",
        "registry": registry_identity,
        "migration_commit": migration_commit,
        "records": records,
    }
