#!/usr/bin/env python3
"""Collect a balanced multi-seed promoted evaluation matrix."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.matrix import (
    aggregate_seeded_evaluation_metrics,
    collect_prompt_metric_rows,
    collect_threshold_metric_rows,
)
from novelty_distill.provenance import repository_commit
from novelty_distill.training.provenance import atomic_json, file_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="METHOD:SEED=PATH")
    parser.add_argument("--control", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--seeds", required=True, metavar="SEED,...")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--seeds must contain comma-separated integers") from error
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain unique non-negative integers")
    return seeds


def _parse_inputs(values: list[str]) -> dict[str, dict[int, Path]]:
    inputs: dict[str, dict[int, Path]] = {}
    for value in values:
        identity, separator, raw_path = value.partition("=")
        method, seed_separator, raw_seed = identity.rpartition(":")
        if (
            not separator
            or not seed_separator
            or not method.strip()
            or not raw_seed.strip()
            or not raw_path.strip()
        ):
            raise ValueError("each --input must be METHOD:SEED=PATH")
        try:
            seed = int(raw_seed)
        except ValueError as error:
            raise ValueError("input seeds must be integers") from error
        if seed < 0 or seed in inputs.setdefault(method, {}):
            raise ValueError(f"invalid or duplicate input for {method!r} seed {raw_seed!r}")
        inputs[method][seed] = Path(raw_path)
    return inputs


def _parse_controls(values: list[str]) -> dict[str, Path]:
    controls: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or not method.strip() or not raw_path.strip() or method in controls:
            raise ValueError("each --control must be one unique METHOD=PATH")
        controls[method] = Path(raw_path)
    return controls


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation artifact is not an object: {path}")
    return payload


def _atomic_rows(output: Path, rows: tuple[dict[str, float | str], ...]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                json.dump(row, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    input_paths = _parse_inputs(args.input)
    control_paths = _parse_controls(args.control)
    if set(input_paths) & set(control_paths):
        raise ValueError("seeded methods and controls must have distinct names")
    seeded_payloads = {
        method: {seed: _load(path) for seed, path in by_seed.items()}
        for method, by_seed in input_paths.items()
    }
    aggregated = aggregate_seeded_evaluation_metrics(seeded_payloads, expected_seeds=seeds)
    combined = {
        **{method: _load(path) for method, path in control_paths.items()},
        **aggregated["evaluations"],
    }
    prompt_rows = collect_prompt_metric_rows(combined)
    threshold_rows = collect_threshold_metric_rows(combined)
    _atomic_rows(args.output, prompt_rows)
    _atomic_rows(args.threshold_output, threshold_rows)

    inputs = {
        f"{method}:{seed}": file_provenance(path)
        for method, by_seed in sorted(input_paths.items())
        for seed, path in sorted(by_seed.items())
    }
    inputs.update(
        {
            f"control:{method}": file_provenance(path)
            for method, path in sorted(control_paths.items())
        }
    )
    manifest = {
        "schema_version": 1,
        "repository_commit": repository_commit(Path(__file__).resolve().parents[1]),
        "expected_seeds": list(seeds),
        "methods": sorted(input_paths),
        "controls": sorted(control_paths),
        "num_prompts": len({str(row["prompt_id"]) for row in prompt_rows}),
        "num_thresholds": len({str(row["threshold"]) for row in threshold_rows}),
        "seed_summaries": aggregated["seed_summaries"],
        "inputs": inputs,
        "outputs": {
            "prompt_metrics": {
                "path": str(args.output.resolve()),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            "threshold_prompt_metrics": {
                "path": str(args.threshold_output.resolve()),
                "sha256": hashlib.sha256(args.threshold_output.read_bytes()).hexdigest(),
            },
        },
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
