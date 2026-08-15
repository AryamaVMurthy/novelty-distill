#!/usr/bin/env python3
"""Derive valid teacher targets that occupy student-undercovered semantic regions."""

import argparse
import hashlib
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from novelty_distill.data.coverage_residual import (
    CoverageCandidate,
    CoverageResidualParameters,
    StudentProbe,
    build_coverage_residual_artifact,
)
from novelty_distill.training.provenance import atomic_json

Record = TypeVar("Record", bound=BaseModel)


def _load_jsonl(path: Path, model: type[Record]) -> tuple[Record, ...]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid record at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"semantic bank is empty: {path}")
    return tuple(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--student-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kappa", type=float, choices=(8.0, 16.0, 32.0), required=True)
    parser.add_argument("--gamma", type=float, choices=(0.5, 1.0), required=True)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--minimum-quality", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = _load_jsonl(args.teacher_bank, CoverageCandidate)
    probes = _load_jsonl(args.student_bank, StudentProbe)
    parameters = CoverageResidualParameters(
        kappa=args.kappa,
        gamma=args.gamma,
        epsilon=args.epsilon,
        minimum_quality=args.minimum_quality,
    )
    artifact = build_coverage_residual_artifact(
        candidates,
        probes,
        parameters=parameters,
        source_hashes={
            "teacher_bank": hashlib.sha256(args.teacher_bank.read_bytes()).hexdigest(),
            "student_bank": hashlib.sha256(args.student_bank.read_bytes()).hexdigest(),
        },
    )
    atomic_json(args.output, artifact)
    print(f"{len(artifact['targets'])}\t{args.output}")


if __name__ == "__main__":
    main()
