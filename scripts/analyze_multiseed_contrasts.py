#!/usr/bin/env python3
"""Analyze a balanced checkpoint-seed matrix without pseudo-replicating prompts."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.multiseed import analyze_checkpoint_seed_contrasts
from novelty_distill.provenance import repository_commit
from novelty_distill.training.provenance import atomic_json, file_provenance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="METHOD:SEED=PATH")
    parser.add_argument("--control", action="append", required=True, metavar="METHOD=PATH")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation artifact is not an object: {path}")
    return payload


def _parse_inputs(values: list[str]) -> tuple[dict[str, dict[int, Path]], dict[str, Path]]:
    inputs: dict[str, dict[int, Path]] = {}
    provenance: dict[str, Path] = {}
    for value in values:
        identity, separator, raw_path = value.partition("=")
        method, seed_separator, raw_seed = identity.rpartition(":")
        if not separator or not seed_separator or not method or not raw_path:
            raise ValueError("each --input must be METHOD:SEED=PATH")
        try:
            seed = int(raw_seed)
        except ValueError as error:
            raise ValueError("input seeds must be integers") from error
        if seed < 0 or seed in inputs.setdefault(method, {}):
            raise ValueError(f"duplicate or invalid input {identity!r}")
        path = Path(raw_path)
        inputs[method][seed] = path
        provenance[identity] = path
    return inputs, provenance


def _parse_controls(values: list[str]) -> tuple[dict[str, Path], dict[str, Path]]:
    controls: dict[str, Path] = {}
    provenance: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or not method or not raw_path or method in controls:
            raise ValueError("each --control must be one unique METHOD=PATH")
        controls[method] = Path(raw_path)
        provenance[f"control:{method}"] = controls[method]
    return controls, provenance


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Compact K=4 three-seed checkpoint analysis",
        "",
        payload["claim_boundary"],
        "",
    ]
    for metric, contrasts in payload["analysis"]["results"].items():
        lines.extend(
            (
                f"## {metric}",
                "",
                "| Contrast | Mean delta | Seed SD | 95% t CI | Seed deltas |",
                "|---|---:|---:|---:|---|",
            )
        )
        for contrast_id, result in contrasts.items():
            seed_values = ", ".join(
                f"{row['checkpoint_seed']}: {row['mean_difference']:.4f}"
                for row in result["per_seed"]
            )
            lines.append(
                f"| {contrast_id} | {result['mean_difference']:.4f} | "
                f"{result['seed_standard_deviation']:.4f} | "
                f"[{result['t_95_ci_low']:.4f}, {result['t_95_ci_high']:.4f}] | "
                f"{seed_values} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported multiseed contrast configuration")
    input_paths, input_provenance = _parse_inputs(args.input)
    control_paths, control_provenance = _parse_controls(args.control)
    analysis = analyze_checkpoint_seed_contrasts(
        evaluations={
            method: {seed: _load(path) for seed, path in by_seed.items()}
            for method, by_seed in input_paths.items()
        },
        controls={method: _load(path) for method, path in control_paths.items()},
        seeds=tuple(int(value) for value in config["seeds"]),
        contrasts=tuple(config["contrasts"]),
        metrics=tuple(str(value) for value in config["metrics"]),
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    payload = {
        "schema_version": 1,
        "repository_commit": repository_commit(Path(__file__).resolve().parents[1]),
        "study": config["study"],
        "claim_scope": config["claim_scope"],
        "claim_boundary": config["claim_boundary"],
        "config": {
            "path": str(args.config.resolve()),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "inputs": {
            identity: file_provenance(path)
            for identity, path in sorted({**input_provenance, **control_provenance}.items())
        },
        "analysis": analysis,
    }
    atomic_json(args.output, payload)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown)}))


if __name__ == "__main__":
    main()
