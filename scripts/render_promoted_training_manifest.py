#!/usr/bin/env python3
"""Render the exact promoted training submission graph for downstream evaluation."""

import argparse
import json
from pathlib import Path

from novelty_distill.provenance import repository_commit
from novelty_distill.training.provenance import atomic_json
from novelty_distill.training.scale_config import build_promoted_training_manifest


def _integers(value: str, *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"{name} must contain comma-separated integers") from error
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _dependencies(values: list[str]) -> dict[tuple[int, int], str]:
    result: dict[tuple[int, int], str] = {}
    for value in values:
        identity, separator, dependency = value.partition("=")
        raw_index, identity_separator, raw_seed = identity.partition(":")
        if (
            not separator
            or not identity_separator
            or not raw_index
            or not raw_seed
            or not dependency
        ):
            raise ValueError("each --dependency must be INDEX:SEED=SLURM_ID")
        try:
            key = (int(raw_index), int(raw_seed))
        except ValueError as error:
            raise ValueError("dependency indexes and seeds must be integers") from error
        if key in result:
            raise ValueError(f"duplicate dependency mapping {identity!r}")
        result[key] = dependency
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", choices=("main", "replication"), required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--dependency", action="append", required=True)
    parser.add_argument("--target-gate-job-id", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = build_promoted_training_manifest(
        model_profile=args.model_profile,
        train_size=args.train_size,
        promoted_indices=_integers(args.indices, name="--indices"),
        seeds=_integers(args.seeds, name="--seeds"),
        training_dependencies=_dependencies(args.dependency),
    )
    result["repository_commit"] = repository_commit(Path(__file__).resolve().parents[1])
    result["target_gate_job_id"] = args.target_gate_job_id
    if args.output is not None:
        atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
