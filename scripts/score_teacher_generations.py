#!/usr/bin/env python3
"""Score permanent teacher shards with a pinned SGLang quality judge."""

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import yaml

from novelty_distill.evaluation.score_shards import validate_score_shard
from novelty_distill.evaluation.teacher_annotation import (
    JudgeSpec,
    build_quality_judge_payload,
    parse_quality_judge_response,
)
from novelty_distill.generation.sglang import (
    GenerationSpec,
    load_prompt_shard,
    render_generation_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _post(url: str, payload: dict[str, Any], timeout: float) -> Mapping[str, Any]:
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, Mapping):
        raise ValueError("judge returned a non-object response")
    return body


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    generation_payload = yaml.safe_load(args.generation_config.read_text(encoding="utf-8"))
    generation_spec = GenerationSpec.model_validate(generation_payload)
    annotation = yaml.safe_load(args.annotation_config.read_text(encoding="utf-8"))
    judge_spec = JudgeSpec(
        model=annotation["judge_model"],
        revision=annotation["judge_revision"],
    )
    prompts: dict[str, str] = {}
    with args.prompts.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                prompts[row["id"]] = row["student_prompt"]

    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    def score_record(record: Any, *, prompt_id: str) -> dict[str, Any]:
        payload = build_quality_judge_payload(
            prompt=render_generation_prompt(prompts[prompt_id], generation_spec),
            response=record.text,
            spec=judge_spec,
        )
        quality = parse_quality_judge_response(_post(endpoint, payload, args.timeout), judge_spec)
        return {
            "prompt_id": prompt_id,
            "sample_index": record.sample_index,
            "text": record.text,
            "finish_reason": record.finish_reason,
            "completion_tokens": record.completion_tokens,
            **quality.model_dump(mode="json"),
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for generation_path in sorted(args.generation_dir.glob("*.json")):
            records = load_prompt_shard(generation_path, generation_spec)
            prompt_id = records[0].prompt_id
            if prompt_id not in prompts:
                raise ValueError(f"generation prompt {prompt_id} is absent from the prompt dataset")
            output_path = args.output_dir / generation_path.name
            text_hashes = [hashlib.sha256(record.text.encode()).hexdigest() for record in records]
            if validate_score_shard(
                output_path,
                prompt_id=prompt_id,
                text_hashes=text_hashes,
                judge=judge_spec,
            ):
                completed += 1
                continue

            scored = list(executor.map(partial(score_record, prompt_id=prompt_id), records))
            _atomic_json(
                output_path,
                {
                    "schema_version": 2,
                    "prompt_id": prompt_id,
                    "text_hashes": text_hashes,
                    "judge": judge_spec.model_dump(mode="json"),
                    "records": scored,
                },
            )
            completed += 1
            print(json.dumps({"scored_prompt": prompt_id, "completed": completed}, sort_keys=True))
    if completed == 0:
        raise ValueError(f"no generation shards found in {args.generation_dir}")


if __name__ == "__main__":
    main()
