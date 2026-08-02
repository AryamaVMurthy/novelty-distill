#!/usr/bin/env python3
"""Derive all static teacher views from one scored eight-sample generation set."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from novelty_distill.data.teacher_views import (
    TeacherGeneration,
    build_teacher_target_artifact,
)
from novelty_distill.provenance import repository_commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    git_commit = repository_commit(Path(__file__).resolve().parents[1])

    generations: list[TeacherGeneration] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                generations.append(TeacherGeneration.model_validate_json(line))
    artifact = build_teacher_target_artifact(generations, seed=args.seed)
    artifact["provenance"] = {
        "clustered_input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "git_commit": git_commit,
        "num_generations": len(generations),
        "seed": args.seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f"{len(artifact['targets'])}\t{args.output}")


if __name__ == "__main__":
    main()
