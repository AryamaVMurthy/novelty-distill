#!/usr/bin/env python3
"""Analyze prompt-matched teacher/human geometry from the existing embedding cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.embeddings import (
    EmbeddingCache,
    embedding_cache_fingerprint,
)
from novelty_distill.evaluation.same_prompt_geometry import analyze_same_prompt_geometry
from novelty_distill.evaluation.score_shards import load_score_shard
from novelty_distill.provenance import repository_commit
from novelty_distill.training.provenance import atomic_json, file_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="METHOD=DIR")
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or not method.strip() or not raw_path.strip():
            raise ValueError("each geometry input must be METHOD=DIR")
        if method in result:
            raise ValueError(f"duplicate geometry method {method!r}")
        result[method] = Path(raw_path)
    return result


def _load_texts(directory: Path) -> dict[str, tuple[str, ...]]:
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no score shards found in {directory}")
    result: dict[str, tuple[str, ...]] = {}
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        text_hashes = raw.get("text_hashes") if isinstance(raw, Mapping) else None
        if not isinstance(text_hashes, list) or not text_hashes:
            raise ValueError(f"score shard has no sample-count evidence: {path}")
        payload = load_score_shard(path, samples_per_prompt=len(text_hashes))
        prompt_id = str(payload["prompt_id"])
        if prompt_id in result:
            raise ValueError(f"duplicate score prompt ID {prompt_id!r} in {directory}")
        result[prompt_id] = tuple(str(record["text"]) for record in payload["records"])
    sample_counts = {len(values) for values in result.values()}
    if len(sample_counts) != 1:
        raise ValueError(f"score directory has inconsistent sample counts: {directory}")
    return result


def _tree_provenance(directory: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(directory.glob("*.json")):
        content = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        count += 1
        total_bytes += len(content)
    if count == 0:
        raise ValueError(f"no score shards found in {directory}")
    return {
        "path": str(directory.resolve()),
        "num_shards": count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _interval(metric: Mapping[str, float]) -> str:
    return (
        f"{metric['estimate']:.4f} "
        f"[{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]"
    )


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Same-prompt embedding geometry",
        "",
        "This paper-inspired diagnostic is secondary and does not alter the frozen primary "
        "outcomes.",
        "Positive teacher-minus-human affinity means closer to A1 than A3 under the frozen "
        "Qwen3 embedding representation.",
        "",
        "| Method | Cosine to teacher | Cosine to human | Teacher − human | Within-method cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, analysis in result["methods"].items():
        metrics = analysis["metrics"]
        lines.append(
            f"| {method} | {_interval(metrics['teacher_cosine_mean'])} | "
            f"{_interval(metrics['human_cosine_mean'])} | "
            f"{_interval(metrics['teacher_minus_human_affinity'])} | "
            f"{_interval(metrics['within_method_pair_cosine'])} |"
        )
    lines.extend(
        [
            "",
            "All intervals are prompt-level paired bootstrap intervals. Higher within-method "
            "cosine "
            "means greater geometric concentration, not higher scientific quality.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    inputs = _parse_inputs(args.input)
    annotation = yaml.safe_load(args.annotation_config.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.analysis_config.read_text(encoding="utf-8"))
    if config.get("status") != "secondary_descriptive":
        raise ValueError("same-prompt geometry must remain secondary_descriptive")

    model_id = str(annotation["embedding_model"])
    revision = str(annotation["embedding_revision"])
    max_length = int(annotation["embedding_max_length"])
    batch_size = int(annotation["embedding_batch_size"])
    fingerprint = embedding_cache_fingerprint(
        model_id=model_id,
        revision=revision,
        max_length=max_length,
        batch_size=batch_size,
    )
    manifest_path = args.embedding_cache_dir / fingerprint / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"embedding cache manifest is missing: {manifest_path}")
    cache = EmbeddingCache(
        args.embedding_cache_dir,
        model_id=model_id,
        revision=revision,
        max_length=max_length,
        batch_size=batch_size,
    )
    instruction = str(annotation["embedding_instruction"])
    texts = {method: _load_texts(path) for method, path in inputs.items()}
    vectors: dict[str, dict[str, tuple[tuple[float, ...], ...]]] = {}
    for method, prompt_texts in texts.items():
        vectors[method] = {}
        for prompt_id, responses in prompt_texts.items():
            prompt_vectors: list[tuple[float, ...]] = []
            for response in responses:
                embedded_text = f"Instruct: {instruction}\nQuery: {response}"
                vector = cache.get(embedded_text)
                if vector is None:
                    raise ValueError(
                        f"cached embedding is missing for {method}/{prompt_id}; "
                        "the CPU analysis will not launch new inference"
                    )
                prompt_vectors.append(tuple(vector))
            vectors[method][prompt_id] = tuple(prompt_vectors)

    bootstrap = config["bootstrap"]
    analysis_config = config["analysis"]
    result = analyze_same_prompt_geometry(
        vectors,
        human_method=str(analysis_config["human_method"]),
        teacher_method=str(analysis_config["teacher_method"]),
        base_method=str(analysis_config["base_method"]),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
        confidence_level=float(bootstrap["confidence_level"]),
    )
    result["source"] = config["source"]
    result["repository_commit"] = repository_commit(Path(__file__).resolve().parents[1])
    result["embedding"] = {
        "model": model_id,
        "revision": revision,
        "instruction": instruction,
        "max_length": max_length,
        "cache_fingerprint": fingerprint,
        "cache_manifest": file_provenance(manifest_path),
    }
    result["inputs"] = {
        method: _tree_provenance(path) for method, path in sorted(inputs.items())
    }
    result["config"] = file_provenance(args.analysis_config)
    atomic_json(args.output, result)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
