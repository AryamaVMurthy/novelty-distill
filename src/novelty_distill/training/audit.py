"""Fail-closed audit of a complete, comparable training matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from novelty_distill.config import BaselineConfig, load_baseline_registry
from novelty_distill.generation.sglang import model_artifact_identity
from novelty_distill.training.provenance import file_provenance


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load training metadata {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"training metadata is not an object: {path}")
    return payload


def _required_string(payload: Mapping[str, Any], field: str, *, source: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"training metadata has no {field}: {source}")
    return value


def _required_positive_int(payload: Mapping[str, Any], field: str, *, source: Path) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"training metadata has invalid {field}: {source}")
    return value


def _sha256_sequence(values: list[str]) -> str:
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact_path(payload: Mapping[str, Any], metadata_path: Path) -> Path:
    raw_path = payload.get("final_dir", payload.get("output_dir"))
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"training metadata has no deployable artifact path: {metadata_path}")
    artifact = Path(raw_path).resolve(strict=True)
    run_root = metadata_path.parent.resolve(strict=True)
    if not artifact.is_relative_to(run_root):
        raise ValueError(f"model artifact escapes its training directory: {artifact}")
    return artifact


def _input_provenance(payload: Mapping[str, Any], metadata_path: Path) -> dict[str, Any]:
    artifacts = payload.get("training_artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("input"), Mapping):
        raise ValueError(f"training metadata has no input provenance: {metadata_path}")
    recorded = dict(artifacts["input"])
    raw_path = recorded.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"training input provenance has no path: {metadata_path}")
    observed = file_provenance(Path(raw_path))
    if recorded != observed:
        raise ValueError(f"training input provenance does not match live bytes: {metadata_path}")
    return observed


def _validate_registry_provenance(
    payload: Mapping[str, Any], metadata_path: Path, registry_provenance: Mapping[str, Any]
) -> None:
    artifacts = payload.get("training_artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("registry"), Mapping):
        raise ValueError(f"training metadata has no registry provenance: {metadata_path}")
    recorded = artifacts["registry"]
    if (
        recorded.get("sha256") != registry_provenance["sha256"]
        or recorded.get("size_bytes") != registry_provenance["size_bytes"]
    ):
        raise ValueError(f"training registry provenance is stale: {metadata_path}")


def _validate_registry_fields(
    baseline: BaselineConfig, payload: Mapping[str, Any], metadata_path: Path
) -> None:
    expected = {
        "backend": baseline.backend,
        "target_view": baseline.target_view,
        "divergence": baseline.divergence,
    }
    for field, value in expected.items():
        if payload.get(field, "none") != value:
            raise ValueError(
                f"training metadata {field} disagrees with registry for "
                f"{baseline.id}: {metadata_path}"
            )
    if baseline.backend in {"trl_gkd", "opsd"}:
        for field in ("lmbda", "beta"):
            if payload.get(field) != getattr(baseline, field):
                raise ValueError(
                    f"training metadata {field} disagrees with registry for "
                    f"{baseline.id}: {metadata_path}"
                )


def _completion_evidence(
    baseline: BaselineConfig, payload: Mapping[str, Any], metadata_path: Path
) -> dict[str, Any]:
    max_steps = _required_positive_int(payload, "max_steps", source=metadata_path)
    if baseline.backend == "distillm":
        log_audit = payload.get("log_audit")
        if (
            not isinstance(log_audit, Mapping)
            or log_audit.get("last_global_step") != max_steps
            or log_audit.get("logged_training_steps") != max_steps
            or not isinstance(log_audit.get("validation_checks"), int)
            or log_audit["validation_checks"] <= 0
        ):
            raise ValueError(
                f"training metadata has no completed training evidence: {metadata_path}"
            )
        return {
            "kind": "official_log_audit",
            "last_global_step": log_audit["last_global_step"],
            "logged_training_steps": log_audit["logged_training_steps"],
            "validation_checks": log_audit["validation_checks"],
        }

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError(f"training metadata has no completed training evidence: {metadata_path}")
    runtime = metrics.get("train_runtime")
    if not isinstance(runtime, int | float) or isinstance(runtime, bool) or runtime <= 0:
        raise ValueError(
            f"training metadata has invalid completed training evidence: {metadata_path}"
        )
    return {
        "kind": "trainer_metrics",
        "metric_keys": sorted(str(key) for key in metrics),
        "train_runtime_seconds": runtime,
    }


def _audit_run(
    baseline: BaselineConfig,
    metadata_path: Path,
    *,
    registry_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_metadata = metadata_path.resolve(strict=True)
    payload = _load_object(resolved_metadata)
    if payload.get("baseline_id") != baseline.id:
        raise ValueError(
            f"training metadata baseline ID disagrees with mapping for {baseline.id}: "
            f"{resolved_metadata}"
        )
    _validate_registry_fields(baseline, payload, resolved_metadata)
    _validate_registry_provenance(payload, resolved_metadata, registry_provenance)

    example_ids = payload.get("example_ids")
    if (
        not isinstance(example_ids, list)
        or not example_ids
        or not all(isinstance(value, str) and value for value in example_ids)
        or len(example_ids) != len(set(example_ids))
    ):
        raise ValueError(f"training metadata has invalid example IDs: {resolved_metadata}")
    training_rows = _required_positive_int(payload, "training_rows", source=resolved_metadata)
    expected_rows = len(example_ids) * (4 if baseline.target_view == "diverse4" else 1)
    if training_rows != expected_rows:
        raise ValueError(
            f"training row count disagrees with target view for {baseline.id}: "
            f"expected {expected_rows}, found {training_rows}"
        )

    model = payload.get("model", payload.get("student_model"))
    revision = payload.get("revision", payload.get("student_revision"))
    if not isinstance(model, str) or not model or not isinstance(revision, str) or not revision:
        raise ValueError(f"training metadata has no student identity: {resolved_metadata}")
    input_provenance = _input_provenance(payload, resolved_metadata)
    artifact = _artifact_path(payload, resolved_metadata)
    seed = _required_positive_int(payload, "seed", source=resolved_metadata)
    exposures = _required_positive_int(
        payload, "optimizer_example_exposures", source=resolved_metadata
    )
    max_steps = _required_positive_int(payload, "max_steps", source=resolved_metadata)
    completion_evidence = _completion_evidence(baseline, payload, resolved_metadata)
    dataset_revision = _required_string(payload, "dataset_revision", source=resolved_metadata)
    git_commit = _required_string(payload, "git_commit", source=resolved_metadata)
    slurm_job_id = _required_string(payload, "slurm_job_id", source=resolved_metadata)

    common = {
        "dataset_revision": dataset_revision,
        "example_count": len(example_ids),
        "example_ids_sha256": _sha256_sequence(example_ids),
        "input_sha256": input_provenance["sha256"],
        "input_size_bytes": input_provenance["size_bytes"],
        "model": model,
        "revision": revision,
        "seed": seed,
        "optimizer_example_exposures": exposures,
    }
    record = {
        "metadata_path": str(resolved_metadata),
        "metadata_sha256": file_provenance(resolved_metadata)["sha256"],
        "backend": baseline.backend,
        "trajectory_source_declared": baseline.trajectory_source,
        "trajectory_source_recorded": payload.get("trajectory_source"),
        "target_view": baseline.target_view,
        "divergence": baseline.divergence,
        "teacher_context": baseline.teacher_context,
        "lmbda": baseline.lmbda,
        "beta": baseline.beta,
        "training_rows": training_rows,
        "optimizer_example_exposures": exposures,
        "max_steps": max_steps,
        "completion_evidence": completion_evidence,
        "seed": seed,
        "git_commit": git_commit,
        "slurm_job_id": slurm_job_id,
        "artifact_path": str(artifact),
        "artifact_identity": model_artifact_identity(artifact),
    }
    return record, common


def audit_training_matrix(
    registry_path: Path,
    metadata_paths: Mapping[str, Path],
    *,
    required_baseline_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate every runnable training baseline and return a byte-bound audit."""

    registry = load_baseline_registry(registry_path)
    all_runnable = {
        baseline.id: baseline
        for baseline in registry.baselines
        if baseline.backend != "evaluation" and baseline.execution_status == "runnable"
    }
    if required_baseline_ids is None:
        runnable = all_runnable
        scope = "complete_registry"
    else:
        requested = tuple(required_baseline_ids)
        if (
            not requested
            or len(requested) != len(set(requested))
            or any(not value for value in requested)
        ):
            raise ValueError("required baseline IDs must be non-empty and unique")
        if unavailable := sorted(set(requested) - set(all_runnable)):
            raise ValueError(f"required baselines are not runnable registry entries: {unavailable}")
        runnable = {baseline_id: all_runnable[baseline_id] for baseline_id in requested}
        scope = "explicit_runnable_subset"
    provided = set(metadata_paths)
    expected = set(runnable)
    if missing := sorted(expected - provided):
        raise ValueError(f"training matrix is missing metadata for: {missing}")
    if extra := sorted(provided - expected):
        raise ValueError(f"training matrix has undeclared metadata for: {extra}")

    registry_provenance = file_provenance(registry_path)
    runs: dict[str, Any] = {}
    common_contract: dict[str, Any] | None = None
    for baseline_id in sorted(runnable):
        record, common = _audit_run(
            runnable[baseline_id],
            metadata_paths[baseline_id],
            registry_provenance=registry_provenance,
        )
        if common_contract is None:
            common_contract = common
        elif common != common_contract:
            differing = sorted(
                key for key in common_contract if common[key] != common_contract[key]
            )
            raise ValueError(
                f"training comparability contract differs for {baseline_id}: {differing}"
            )
        runs[baseline_id] = record

    fail_closed = [
        {"id": baseline.id, "reason": baseline.non_executable_reason}
        for baseline in registry.baselines
        if baseline.backend != "evaluation" and baseline.execution_status == "fail_closed"
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "scope": scope,
        "registry": registry_provenance,
        "executable_baseline_count": len(runnable),
        "executable_baseline_ids": sorted(runnable),
        "fail_closed_baselines": fail_closed,
        "common_contract": common_contract,
        "runs": runs,
    }
