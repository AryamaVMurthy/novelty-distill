#!/usr/bin/env python3
"""Create reproducible findings for frozen teacher target views and clusters."""

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from novelty_distill.data.teacher_views import TeacherGeneration
from novelty_distill.evaluation.score_shards import load_score_shard
from novelty_distill.evaluation.teacher_target_analysis import (
    render_teacher_target_markdown,
    summarize_teacher_targets,
)
from novelty_distill.provenance import repository_commit
from novelty_distill.training.provenance import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--clustered", type=Path, required=True)
    parser.add_argument("--cluster-metadata", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    git_commit = repository_commit(Path(__file__).resolve().parents[1])
    if args.expected_prompts <= 0:
        raise ValueError("expected-prompts must be positive")
    score_paths = sorted(args.score_dir.glob("*.json"))
    if len(score_paths) != args.expected_prompts:
        raise ValueError(
            f"expected {args.expected_prompts} scores in {args.score_dir}, found {len(score_paths)}"
        )
    scores = tuple(load_score_shard(path, samples_per_prompt=8) for path in score_paths)
    generations = tuple(
        TeacherGeneration.model_validate_json(line)
        for line in args.clustered.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    metadata = json.loads(args.cluster_metadata.read_text(encoding="utf-8"))
    if not isinstance(targets, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("target and cluster metadata artifacts must be objects")
    summary = summarize_teacher_targets(
        generations=generations,
        score_payloads=scores,
        target_artifact=targets,
        cluster_metadata=metadata,
        seed=args.seed,
    )
    summary["schema_version"] = 1
    summary["git_commit"] = git_commit
    summary["inputs"] = {
        "scores": {"path": str(args.score_dir), "tree_sha256": _tree_hash(score_paths)},
        "clustered": {"path": str(args.clustered), "sha256": _sha256(args.clustered)},
        "cluster_metadata": {
            "path": str(args.cluster_metadata),
            "sha256": _sha256(args.cluster_metadata),
        },
        "targets": {"path": str(args.targets), "sha256": _sha256(args.targets)},
    }
    atomic_json(args.output, summary)
    _atomic_text(args.markdown, render_teacher_target_markdown(summary))
    print(
        json.dumps(
            {
                "num_prompts": summary["num_prompts"],
                "output": str(args.output),
                "markdown": str(args.markdown),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
