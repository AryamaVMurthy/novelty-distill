#!/usr/bin/env python3
"""Run a resumable teacher sampling matrix against one SGLang server."""

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.teacher_calibration import TeacherCalibrationStudy
from novelty_distill.generation.sglang import (
    generate_prompt,
    generation_fingerprint,
    load_prompts,
    pending_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    study = TeacherCalibrationStudy.model_validate(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    prompts = load_prompts(args.input)
    if len(prompts) != study.prompt_count:
        raise ValueError(
            f"study requires {study.prompt_count} prompts, input contains {len(prompts)}"
        )
    manifest = {
        "schema_version": 1,
        "study": study.name,
        "study_config_sha256": _sha256(args.config),
        "input_sha256": _sha256(args.input),
        "prompt_ids": [prompt.id for prompt in prompts],
        "reference_condition": study.reference_condition,
        "conditions": {
            condition.id: {
                "generation_fingerprint": generation_fingerprint(condition.generation),
                "generation": condition.generation.model_dump(mode="json"),
            }
            for condition in study.conditions
        },
    }
    manifest_path = args.output_dir / "study-manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"calibration study changed for existing {manifest_path}")
    else:
        _atomic_json(manifest_path, manifest)

    endpoint = args.base_url.rstrip("/")
    for condition in study.conditions:
        condition_dir = args.output_dir / condition.id
        shards = condition_dir / "shards"
        config_path = condition_dir / "generation-config.json"
        config_payload = condition.generation.model_dump(mode="json")
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config_payload:
                raise ValueError(f"generation config changed for {condition.id}")
        else:
            _atomic_json(config_path, config_payload)
        remaining = pending_prompts(prompts, shards, condition.generation)
        print(
            json.dumps(
                {
                    "condition": condition.id,
                    "total": len(prompts),
                    "pending": len(remaining),
                },
                sort_keys=True,
            )
        )

        def run(
            prompt: Any,
            generation: Any = condition.generation,
            output_dir: Path = shards,
        ) -> str:
            generate_prompt(
                prompt,
                generation,
                output_dir,
                base_url=endpoint,
                timeout=args.timeout,
            )
            return prompt.id

        with ThreadPoolExecutor(max_workers=study.concurrency) as executor:
            for completed, _prompt_id in enumerate(executor.map(run, remaining), start=1):
                print(
                    json.dumps(
                        {
                            "condition": condition.id,
                            "completed": completed,
                            "pending_at_start": len(remaining),
                        },
                        sort_keys=True,
                    )
                )


if __name__ == "__main__":
    main()
