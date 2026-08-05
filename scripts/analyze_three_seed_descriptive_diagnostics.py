#!/usr/bin/env python3
"""Build automatic descriptive diagnostics from the complete compact matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.descriptive_diagnostics import (
    analyze_three_seed_descriptive_diagnostics,
)
from novelty_distill.provenance import repository_commit
from novelty_distill.training.provenance import atomic_json, file_provenance

MappingLike = dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--seed17-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _expected_hashes(
    *, manifest: MappingLike, seed17_summary: MappingLike, required_keys: set[str]
) -> dict[str, str]:
    manifest_inputs = manifest.get("inputs")
    summary_methods = seed17_summary.get("methods")
    if not isinstance(manifest_inputs, dict) or not isinstance(summary_methods, dict):
        raise ValueError("input manifest or seed-17 summary has invalid structure")
    expected = {
        str(identity): str(provenance["sha256"]) for identity, provenance in manifest_inputs.items()
    }
    expected["control:A1"] = str(summary_methods["A1"]["input_sha256"])
    if set(expected) != required_keys:
        missing = sorted(required_keys - set(expected))
        unexpected = sorted(set(expected) - required_keys)
        raise ValueError(f"input manifest mismatch; missing={missing}, unexpected={unexpected}")
    for method, summary in summary_methods.items():
        identity = f"control:{method}" if method in {"A0", "A1"} else f"{method}:17"
        if identity in expected and str(summary["input_sha256"]) != expected[identity]:
            raise ValueError(f"seed-17 hash disagreement for {identity}")
    return expected


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _judge_row(name: str, summary: MappingLike) -> str:
    dimensions = summary["dimensions"]
    feasibility = dimensions["student_feasibility_mean"]
    soundness = dimensions["student_soundness_mean"]
    relevance = dimensions["student_relevance_mean"]
    clarity = dimensions["student_clarity_mean"]
    compliance = dimensions["student_instruction_compliance_mean"]
    return (
        f"| {name} | {_fmt(feasibility['mean'])} | "
        f"{_fmt(feasibility['prompt_ceiling_rate'])} | {_fmt(soundness['mean'])} | "
        f"{_fmt(soundness['prompt_ceiling_rate'])} | "
        f"{_fmt(relevance['prompt_near_ceiling_rate'])} | "
        f"{_fmt(clarity['prompt_near_ceiling_rate'])} | "
        f"{_fmt(compliance['prompt_near_ceiling_rate'])} | "
        f"{_fmt(summary['completion_tokens_mean'], 1)} | "
        f"{_fmt(summary['length_stop_rate'], 6)} |"
    )


def _judge_length_row(name: str, summary: MappingLike) -> str:
    dimensions = summary["dimensions"]
    feasibility = dimensions["student_feasibility_mean"]
    soundness = dimensions["student_soundness_mean"]
    return (
        f"| {name} | {_fmt(feasibility['prompt_low_rate'])} | "
        f"{_fmt(soundness['prompt_low_rate'])} | "
        f"{_fmt(feasibility['prompt_length_pearson'])} | "
        f"{_fmt(soundness['prompt_length_pearson'])} |"
    )


def _render_markdown(payload: MappingLike) -> str:
    analysis = payload["analysis"]
    judge = analysis["judge_prompt_mean_diagnostics"]
    curves = analysis["semantic_method_curves"]["quality_qualified_semantic_yield"]
    lines = [
        "# Three-seed automatic descriptive diagnostics",
        "",
        f"Producer commit: `{payload['repository_commit']}`",
        "",
        analysis["claim_boundary"],
        "",
        "This report deliberately leaves the human-calibrated semantic boundary unresolved. "
        "All semantic tables below are threshold sensitivity diagnostics, not novelty scores.",
        "",
        "## Judge score-distribution audit",
        "",
        "Rates are computed over the 1,658 prompt-level K=4 means. A ceiling rate therefore "
        "means all four answers received 5/5 on that dimension for a prompt; it is not an "
        "individual-answer ceiling rate.",
        "",
        "| Method | Feas. mean | Feas. ceiling | Sound. mean | Sound. ceiling | "
        "Relevance >=4.75 | Clarity >=4.75 | Compliance >=4.75 | Tokens | Length stop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in judge["fixed_controls"].items():
        lines.append(_judge_row(f"{method} (fixed)", result))
    for method, result in judge["trained_methods"].items():
        lines.append(_judge_row(method, result["across_seeds"]))

    lines.extend(
        (
            "",
            "Prompt-low rates use a K=4 prompt mean at or below 3/5. Length correlations are "
            "within-run Pearson correlations between the prompt's mean completion length and "
            "the prompt-level dimension mean.",
            "",
            "| Method | Feasibility <=3 | Soundness <=3 | Feas.-length r | Sound.-length r |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for method, result in judge["fixed_controls"].items():
        lines.append(_judge_length_row(f"{method} (fixed)", result))
    for method, result in judge["trained_methods"].items():
        lines.append(_judge_length_row(method, result["across_seeds"]))

    methods = [*judge["fixed_controls"], *judge["trained_methods"]]
    lines.extend(
        (
            "",
            "## Quality-qualified semantic yield across all thresholds",
            "",
            "Each trained entry is the mean of checkpoint seeds 17/29/43. A0/A1 are fixed "
            "seed-17 generation realizations.",
            "",
            "| Threshold | " + " | ".join(methods) + " |",
            "|---:|" + "---:|" * len(methods),
        )
    )
    for threshold in analysis["thresholds"]:
        lines.append(
            f"| {threshold} | "
            + " | ".join(_fmt(curves[method][threshold]["mean"]) for method in methods)
            + " |"
        )

    lines.extend(
        (
            "",
            "## Contrast stability over the eight embedding thresholds",
            "",
            "`Oriented range` is positive in the configured desirable direction (higher for "
            "coverage/recall/precision, lower for ClusterJSD). Direction is descriptive and "
            "does not make every teacher-fidelity metric scientific value.",
            "",
            "| Metric | Contrast | Favorable thresholds | Direction | Oriented range |",
            "|---|---|---:|---|---:|",
        )
    )
    for contrast, metrics in analysis["semantic_contrast_stability"].items():
        for metric, result in metrics.items():
            lines.append(
                f"| {metric} | {contrast} | {result['favorable_threshold_count']}/"
                f"{len(analysis['thresholds'])} | {result['stable_direction']} | "
                f"[{result['oriented_delta_min']:.4f}, {result['oriented_delta_max']:.4f}] |"
            )
    lines.extend(
        (
            "",
            "## Main descriptive findings",
            "",
            "- B2a has more raw student embedding clusters than A0 at all eight thresholds, "
            "but lower quality-qualified yield at all eight. Its extra clusters therefore do "
            "not represent useful breadth under the automatic quality gate.",
            "- C1 has fewer raw clusters than B2b at all eight thresholds but higher "
            "quality-qualified yield at all eight. The central hard-KD failure is quality, not "
            "a simple absence of string or embedding variation.",
            "- C2 is below C1 on both raw clusters and quality-qualified yield at all eight "
            "thresholds, while having higher teacher-mode precision and lower ClusterJSD at "
            "all eight. In this embedding geometry, reverse KL is more teacher-concentrated "
            "and less broad.",
            "- D1 versus C1 is mixed across thresholds on every semantic diagnostic; there is "
            "no stable automatic on-policy forward-KL advantage.",
            "- D2 versus C2 is mixed on qualified yield and raw clusters, but has lower "
            "teacher-mode precision and higher ClusterJSD at every threshold. The current "
            "on-policy reverse-KL recipe does not improve teacher-distribution fidelity.",
            "- D2 is below D1 on quality-qualified yield and quality-adjusted coverage at all "
            "eight thresholds. This is a robust automatic pattern, not a human-validated "
            "semantic-diversity conclusion.",
            "- Prompt-level soundness-length correlations are near zero for all six trained "
            "methods. Feasibility-length correlations are small (largest for D1), so response "
            "length alone does not explain the hard-KD quality collapse.",
            "",
            "## Interpretation boundary",
            "",
            "- Stable behavior across thresholds is stronger descriptive evidence than an "
            "effect at 0.94 alone, but it still inherits the embedding model and K=4 sample.",
            "- Relevance, clarity, and compliance remain strongly compressed; feasibility and "
            "soundness carry most of the automatic judge discrimination.",
            "- No row establishes literature-grounded novelty or human semantic equivalence.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported descriptive diagnostic config")
    seeds = tuple(int(seed) for seed in config["seeds"])
    trained_filenames = {
        method: {seed: template.format(seed=seed) for seed in seeds}
        for method, template in config["trained_methods"].items()
    }
    control_filenames = dict(config["fixed_controls"])
    required_keys = {
        *(f"{method}:{seed}" for method in trained_filenames for seed in seeds),
        *(f"control:{method}" for method in control_filenames),
    }
    expected = _expected_hashes(
        manifest=_load(args.input_manifest),
        seed17_summary=_load(args.seed17_summary),
        required_keys=required_keys,
    )
    input_paths = {
        **{
            f"{method}:{seed}": args.raw_dir / filename
            for method, by_seed in trained_filenames.items()
            for seed, filename in by_seed.items()
        },
        **{
            f"control:{method}": args.raw_dir / filename
            for method, filename in control_filenames.items()
        },
    }
    provenance = {identity: file_provenance(path) for identity, path in input_paths.items()}
    for identity, details in provenance.items():
        if details["sha256"] != expected[identity]:
            raise ValueError(f"input hash mismatch for {identity}")
    evaluations = {
        method: {seed: _load(args.raw_dir / filename) for seed, filename in by_seed.items()}
        for method, by_seed in trained_filenames.items()
    }
    controls = {
        method: _load(args.raw_dir / filename) for method, filename in control_filenames.items()
    }
    analysis = analyze_three_seed_descriptive_diagnostics(
        evaluations=evaluations,
        controls=controls,
        seeds=seeds,
        contrasts=tuple(config["contrasts"]),
        semantic_metrics=dict(config["semantic_metrics"]),
    )
    if analysis["claim_boundary"] != config["claim_boundary"]:
        raise ValueError("configured claim boundary does not match analyzer policy")
    payload = {
        "schema_version": 1,
        "repository_commit": repository_commit(Path(__file__).resolve().parents[1]),
        "study": config["study"],
        "claim_scope": config["claim_scope"],
        "config": {
            "path": str(args.config.resolve()),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "inputs": dict(sorted(provenance.items())),
        "analysis": analysis,
    }
    atomic_json(args.output, payload)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown)}))


if __name__ == "__main__":
    main()
