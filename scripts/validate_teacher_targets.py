#!/usr/bin/env python3
"""Validate target IDs, frozen-view reconstruction, and source provenance."""

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from novelty_distill.data.teacher_views import (
    TeacherGeneration,
    validate_teacher_target_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--clustered", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_prompt_ids(path: Path) -> set[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt_id = payload.get("id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"invalid prompt ID at {path}:{line_number}")
            ids.append(prompt_id)
    if not ids:
        raise ValueError(f"no prompts found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate prompt IDs in {path}")
    return set(ids)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
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
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    if args.expected_prompts is not None and args.expected_prompts <= 0:
        raise ValueError("expected-prompts must be positive")
    prompt_ids = _load_prompt_ids(args.prompts)
    if args.expected_prompts is not None and len(prompt_ids) != args.expected_prompts:
        raise ValueError(
            f"expected {args.expected_prompts} prompts in {args.prompts}, found {len(prompt_ids)}"
        )
    generations = tuple(
        TeacherGeneration.model_validate_json(line)
        for line in args.clustered.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    payload = json.loads(args.targets.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"teacher-target artifact is not an object: {args.targets}")
    summary = validate_teacher_target_artifact(
        payload,
        generations=generations,
        seed=args.seed,
        expected_prompt_ids=prompt_ids,
    )
    clustered_sha256 = _sha256(args.clustered)
    expected_provenance = {
        "clustered_input_sha256": clustered_sha256,
        "num_generations": len(generations),
        "seed": args.seed,
    }
    if payload.get("provenance") != expected_provenance:
        raise ValueError("teacher-target provenance does not match its clustered input")
    manifest: dict[str, object] = {
        "schema_version": 1,
        **summary,
        "targets": {"path": str(args.targets), "sha256": _sha256(args.targets)},
        "clustered": {"path": str(args.clustered), "sha256": clustered_sha256},
        "prompts": {"path": str(args.prompts), "sha256": _sha256(args.prompts)},
    }
    if args.output is not None:
        _atomic_json(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
