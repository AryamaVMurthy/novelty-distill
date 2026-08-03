"""Deterministic W&B export without making online tracking a training dependency."""

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_CONFIG_KEYS = (
    "schema_version",
    "baseline_id",
    "backend",
    "model",
    "revision",
    "teacher_model",
    "teacher_revision",
    "student_thinking",
    "divergence",
    "drkl_gamma",
    "lmbda",
    "beta",
    "trajectory_source",
    "target_view",
    "dataset_revision",
    "training_rows",
    "optimizer_example_exposures",
    "max_steps",
    "save_steps",
    "seed",
    "slurm_job_id",
    "git_commit",
    "repository_commit",
    "run_spec",
    "training_artifacts",
    "num_prompts",
    "teacher_samples_per_prompt",
    "student_samples_per_prompt",
    "primary_cosine_threshold",
    "semantic_clustering",
    "embedding",
    "inputs",
    "source",
    "protocol_hash",
    "human_validation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower()
    return cleaned[:80] or "record"


def _record_identity(path: Path, payload: Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(payload.get("baseline_id"), str):
        return "training", str(payload["baseline_id"])
    if isinstance(payload.get("prompt_metrics"), Mapping):
        return "evaluation", path.stem
    if isinstance(payload.get("results"), list | tuple):
        return "analysis", path.stem
    if isinstance(payload.get("methods"), Mapping):
        return "research-taste", path.stem
    return "metadata", path.stem


def _config(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in _CONFIG_KEYS if key in payload}


def _numeric_items(payload: Mapping[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for container_name in ("metrics", "overall"):
        raw = payload.get(container_name)
        if isinstance(raw, Mapping):
            for name, value in raw.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    numeric = float(value)
                    if math.isfinite(numeric):
                        metrics[str(name)] = value
    for name in (
        "num_prompts",
        "training_rows",
        "optimizer_example_exposures",
        "max_steps",
        "teacher_samples_per_prompt",
        "student_samples_per_prompt",
    ):
        value = payload.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                metrics[name] = value
    return metrics


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_manifest(
    path: Path, *, project: str, entity: str | None, mode: str
) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "project": project,
            "entity": entity,
            "mode": mode,
            "records": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError(f"invalid W&B export manifest: {path}")
    expected = {"project": project, "entity": entity, "mode": mode}
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"W&B export manifest destination changed: {actual} != {expected}")
    return payload


def export_wandb_records(
    paths: Sequence[Path],
    *,
    manifest_path: Path,
    wandb_module: Any,
    project: str,
    entity: str | None,
    mode: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Export small immutable JSON artifacts and skip byte-identical repeats."""

    if mode not in {"offline", "online", "disabled"}:
        raise ValueError("W&B mode must be offline, online, or disabled")
    if not project.strip():
        raise ValueError("W&B project must be non-empty")
    resolved_paths = tuple(path.resolve(strict=True) for path in paths)
    if not resolved_paths or len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("W&B export inputs must be a non-empty unique path list")
    if any(not path.is_file() or path.suffix != ".json" for path in resolved_paths):
        raise ValueError("W&B export inputs must be JSON files")

    manifest = _load_manifest(
        manifest_path, project=project, entity=entity, mode=mode
    )
    completed = {
        (str(record["path"]), str(record["sha256"]))
        for record in manifest["records"]
        if isinstance(record, Mapping) and "path" in record and "sha256" in record
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    skipped = 0
    for path in resolved_paths:
        digest = _sha256(path)
        if (str(path), digest) in completed:
            skipped += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"W&B source must contain a JSON object: {path}")
        kind, logical_name = _record_identity(path, payload)
        run_id = f"{_slug(kind)}-{_slug(logical_name)}-{digest[:12]}"
        init_args: dict[str, Any] = {
            "project": project,
            "id": run_id,
            "name": f"{kind}/{logical_name}",
            "job_type": kind,
            "config": _config(payload),
            "mode": mode,
            "dir": str(run_dir),
            "reinit": True,
        }
        if entity is not None:
            init_args["entity"] = entity
        run = wandb_module.init(**init_args)
        try:
            metrics = _numeric_items(payload)
            run.summary.update(metrics)
            table = wandb_module.Table(
                columns=["metric", "value"],
                data=[[name, value] for name, value in sorted(metrics.items())],
            )
            run.log({"metrics": table})
            artifact = wandb_module.Artifact(
                name=f"{_slug(kind)}-{_slug(logical_name)}-{digest[:12]}",
                type=f"{kind}-metadata",
                metadata={
                    "source_path": str(path),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                },
            )
            artifact.add_file(str(path), name=path.name)
            run.log_artifact(artifact)
        finally:
            run.finish()
        manifest["records"].append(
            {
                "path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "kind": kind,
                "logical_name": logical_name,
                "run_id": run_id,
            }
        )
        _atomic_json(manifest_path, manifest)
        exported += 1
    return {
        "exported": exported,
        "skipped": skipped,
        "manifest": str(manifest_path.resolve()),
        "mode": mode,
    }
