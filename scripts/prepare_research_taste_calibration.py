#!/usr/bin/env python3
"""Create a source-blinded human calibration packet from taste annotation shards."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.research_taste import research_taste_protocol_hash
from novelty_distill.evaluation.taste_calibration import (
    select_blinded_taste_calibration,
)
from novelty_distill.evaluation.taste_shards import load_research_taste_shard
from novelty_distill.generation.sglang import (
    GenerationSpec,
    load_prompts,
    render_generation_prompt,
)
from novelty_distill.provenance import repository_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="METHOD=DIR")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    return parser.parse_args()


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or not method.strip() or not raw_path.strip() or method in inputs:
            raise ValueError(f"invalid or duplicate METHOD=DIR input {value!r}")
        inputs[method] = Path(raw_path)
    return inputs


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


def _load_candidates(
    *,
    method: str,
    directory: Path,
    prompts: dict[str, str],
) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no research-taste shards found for {method}: {directory}")
    for path in paths:
        preview = json.loads(path.read_text(encoding="utf-8"))
        raw_records = preview.get("records") if isinstance(preview, dict) else None
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"research-taste shard has no records: {path}")
        payload = load_research_taste_shard(path, samples_per_prompt=len(raw_records))
        prompt_id = payload["prompt_id"]
        if prompt_id not in prompts:
            raise ValueError(f"taste prompt {prompt_id!r} is absent from the prompt dataset")
        for record in payload["records"]:
            candidates.append(
                {
                    "method": method,
                    "prompt_id": prompt_id,
                    "sample_index": record["sample_index"],
                    "task": prompts[prompt_id],
                    "text": record["text"],
                    "opportunity_pattern": record["opportunity_pattern"],
                    "method_paradigm": record["method_paradigm"],
                }
            )
    return tuple(candidates)


def _instructions() -> str:
    return """# Blinded research-taste annotation

Independently label every JSONL record. Do not consult another annotator or any automatic label.
The method identity is intentionally hidden. Replace both null values with exactly one label.

Opportunity pattern labels:
`puzzle_or_contradiction`, `explanation_gap`, `assumption_or_scope_mismatch`,
`measurement_evidence_gap`, `fragmentation_or_bridge_opportunity`, `failure_or_risk_gap`,
`resource_or_operational_constraint`.

Method paradigm labels:
`explicit_synthesis_or_unification`, `assumption_relaxation_or_scope_extension`,
`failure_mitigation_or_robustification`, `formal_conceptual_derivation`,
`measurement_or_empirical_mapping`, `constructive_artifact_or_system`,
`optimization_search_or_resource_strategy`.

Classify why the work is needed separately from how its contribution is constructed. Several
components do not by themselves imply synthesis. Judge the task and candidate idea, not its topic
or writing style. Preserve every `calibration_id` and do not add or remove records.
"""


def main() -> None:
    args = parse_args()
    inputs = _parse_inputs(args.input)
    generation_spec = GenerationSpec.model_validate(
        yaml.safe_load(args.generation_config.read_text(encoding="utf-8"))
    )
    prompts = {
        prompt.id: render_generation_prompt(prompt.text, generation_spec)
        for prompt in load_prompts(args.prompts)
    }
    selected = select_blinded_taste_calibration(
        tuple(
            candidate
            for method, directory in sorted(inputs.items())
            for candidate in _load_candidates(
                method=method, directory=directory, prompts=prompts
            )
        ),
        size=args.size,
        seed=args.seed,
    )
    packet_content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in selected["packet"]
    )
    key_payload = {
        key: value for key, value in selected.items() if key not in {"packet", "key"}
    }
    key_payload.update(
        {
            "protocol_hash": research_taste_protocol_hash(),
            "repository_commit": repository_commit(Path(__file__).resolve().parents[1]),
            "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "generation_config_sha256": hashlib.sha256(
                args.generation_config.read_bytes()
            ).hexdigest(),
            "inputs": {method: str(path.resolve()) for method, path in sorted(inputs.items())},
            "packet_sha256": hashlib.sha256(packet_content.encode()).hexdigest(),
            "key": selected["key"],
        }
    )
    _atomic_text(args.packet, packet_content)
    _atomic_text(args.key, json.dumps(key_payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(args.instructions, _instructions())


if __name__ == "__main__":
    main()
