#!/usr/bin/env python3
"""Evaluate judged student samples against fixed teacher-anchored semantic modes."""

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from novelty_distill.evaluation.embeddings import (
    embed_texts,
    embedding_cache_fingerprint,
)
from novelty_distill.evaluation.score_shards import load_score_shard
from novelty_distill.evaluation.student_evaluation import (
    anchor_student_clusters,
    summarize_generation_diagnostics,
    summarize_quality_dimensions,
    summarize_student_prompt,
)
from novelty_distill.provenance import repository_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-score-dir", type=Path, required=True)
    parser.add_argument("--student-score-dir", type=Path, required=True)
    parser.add_argument("--training-targets", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--teacher-samples-per-prompt", type=int)
    parser.add_argument("--student-samples-per-prompt", type=int)
    return parser.parse_args()


def _load_scores(directory: Path, *, samples_per_prompt: int) -> dict[str, list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no score shards found in {directory}")
    for path in paths:
        payload = load_score_shard(path, samples_per_prompt=samples_per_prompt)
        ordered = payload["records"]
        prompt_id = str(payload.get("prompt_id", ""))
        if not prompt_id or any(str(record.get("prompt_id")) != prompt_id for record in ordered):
            raise ValueError(f"score shard {path} has inconsistent prompt ids")
        if [int(record["sample_index"]) for record in ordered] != list(range(samples_per_prompt)):
            raise ValueError(f"score shard {path} has non-contiguous sample indices")
        if prompt_id in by_prompt:
            raise ValueError(f"duplicate score prompt id {prompt_id}")
        by_prompt[prompt_id] = ordered
    return by_prompt


def _load_training_texts(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("targets")
    if payload.get("schema_version") != 1 or not isinstance(targets, Mapping):
        raise ValueError(f"invalid teacher-target artifact {path}")
    texts = {
        str(text)
        for views in targets.values()
        if isinstance(views, Mapping)
        for text in views.get("all8", ())
        if isinstance(text, str) and text.strip()
    }
    if not texts:
        raise ValueError(f"teacher-target artifact has no all8 responses: {path}")
    return tuple(sorted(texts))


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.json")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
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


def _mean_metrics(prompt_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    names = tuple(next(iter(prompt_metrics.values())).keys())
    return {
        name: statistics.fmean(float(metrics[name]) for metrics in prompt_metrics.values())
        for name in names
    }


def main() -> None:
    args = parse_args()
    git_commit = repository_commit(Path(__file__).resolve().parents[1])
    if args.samples_per_prompt <= 0:
        raise ValueError("samples per prompt must be positive")
    teacher_samples = args.teacher_samples_per_prompt or args.samples_per_prompt
    student_samples = args.student_samples_per_prompt or args.samples_per_prompt
    if teacher_samples <= 0 or student_samples <= 0:
        raise ValueError("teacher and student samples per prompt must be positive")
    teacher = _load_scores(args.teacher_score_dir, samples_per_prompt=teacher_samples)
    student = _load_scores(args.student_score_dir, samples_per_prompt=student_samples)
    if teacher.keys() != student.keys():
        missing_teacher = sorted(student.keys() - teacher.keys())
        missing_student = sorted(teacher.keys() - student.keys())
        raise ValueError(
            "teacher/student prompt sets differ: "
            f"missing_teacher={missing_teacher[:5]} missing_student={missing_student[:5]}"
        )
    training_texts = _load_training_texts(args.training_targets)
    annotation = yaml.safe_load(args.annotation_config.read_text(encoding="utf-8"))
    thresholds = tuple(float(value) for value in annotation["cosine_thresholds"])
    primary_threshold = float(annotation["cosine_threshold"])
    if primary_threshold not in thresholds:
        thresholds = (*thresholds, primary_threshold)

    response_texts = {
        str(record["text"])
        for records in (*teacher.values(), *student.values())
        for record in records
    }
    all_texts = tuple(sorted(response_texts | set(training_texts)))
    instruction = str(annotation["embedding_instruction"])
    embedding_inputs = [f"Instruct: {instruction}\nQuery: {text}" for text in all_texts]
    embedded = embed_texts(
        embedding_inputs,
        model_id=annotation["embedding_model"],
        revision=annotation["embedding_revision"],
        max_length=int(annotation["embedding_max_length"]),
        batch_size=int(annotation["embedding_batch_size"]),
        cache_dir=args.embedding_cache_dir,
    )
    embedding_by_text = dict(zip(all_texts, embedded, strict=True))

    import torch

    target_tensor = torch.tensor(
        [embedding_by_text[text] for text in training_texts],
        dtype=torch.float32,
        device="cuda",
    )
    nearest_by_prompt: dict[str, tuple[float, ...]] = {}
    for prompt_id, records in student.items():
        student_tensor = torch.tensor(
            [embedding_by_text[str(record["text"])] for record in records],
            dtype=torch.float32,
            device="cuda",
        )
        nearest_by_prompt[prompt_id] = tuple(
            (student_tensor @ target_tensor.T).max(dim=1).values.cpu().tolist()
        )

    metrics_by_threshold: dict[str, dict[str, dict[str, float]]] = {}
    for threshold in thresholds:
        prompt_metrics: dict[str, dict[str, float]] = {}
        for prompt_id in sorted(teacher):
            teacher_records = teacher[prompt_id]
            student_records = student[prompt_id]
            teacher_embeddings = [
                embedding_by_text[str(record["text"])] for record in teacher_records
            ]
            student_embeddings = [
                embedding_by_text[str(record["text"])] for record in student_records
            ]
            teacher_labels, student_labels = anchor_student_clusters(
                teacher_embeddings=teacher_embeddings,
                student_embeddings=student_embeddings,
                threshold=threshold,
            )
            semantic_metrics = summarize_student_prompt(
                teacher_clusters=teacher_labels,
                student_clusters=student_labels,
                student_quality_scores=(
                    float(record["quality_score"]) for record in student_records
                ),
                student_feasibility_scores=(
                    int(record["dimensions"]["feasibility"]) for record in student_records
                ),
                nearest_training_target_similarities=nearest_by_prompt[prompt_id],
            )
            diagnostics = summarize_generation_diagnostics(
                finish_reasons=(str(record["finish_reason"]) for record in student_records),
                completion_tokens=(int(record["completion_tokens"]) for record in student_records),
            )
            dimension_metrics = summarize_quality_dimensions(
                record["dimensions"] for record in student_records
            )
            prompt_metrics[prompt_id] = {
                **semantic_metrics,
                **dimension_metrics,
                **diagnostics,
            }
        metrics_by_threshold[f"{threshold:.3f}"] = prompt_metrics

    primary_key = f"{primary_threshold:.3f}"
    primary_metrics = metrics_by_threshold[primary_key]
    payload = {
        "schema_version": 1,
        "git_commit": git_commit,
        "num_prompts": len(primary_metrics),
        "teacher_samples_per_prompt": teacher_samples,
        "student_samples_per_prompt": student_samples,
        "primary_cosine_threshold": primary_threshold,
        "semantic_clustering": {
            "teacher_partition": "connected_components",
            "student_assignment": "nearest_teacher_if_cosine_at_least_threshold",
            "unmatched_student_partition": "connected_components",
            "teacher_modes_are_fixed_across_methods": True,
        },
        "embedding": {
            "model": annotation["embedding_model"],
            "revision": annotation["embedding_revision"],
            "instruction": instruction,
            "max_length": annotation["embedding_max_length"],
            "cache_fingerprint": embedding_cache_fingerprint(
                model_id=annotation["embedding_model"],
                revision=annotation["embedding_revision"],
                max_length=int(annotation["embedding_max_length"]),
                batch_size=int(annotation["embedding_batch_size"]),
            ),
        },
        "inputs": {
            "teacher_score_sha256": _tree_hash(args.teacher_score_dir),
            "student_score_sha256": _tree_hash(args.student_score_dir),
            "training_targets_sha256": hashlib.sha256(
                args.training_targets.read_bytes()
            ).hexdigest(),
            "training_target_count": len(training_texts),
        },
        "overall": _mean_metrics(primary_metrics),
        "prompt_metrics": primary_metrics,
        "prompt_metrics_by_threshold": metrics_by_threshold,
        "threshold_sensitivity": {
            threshold: _mean_metrics(prompt_metrics)
            for threshold, prompt_metrics in metrics_by_threshold.items()
        },
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
