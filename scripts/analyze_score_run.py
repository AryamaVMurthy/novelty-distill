#!/usr/bin/env python3
"""Create reproducible JSON and Markdown diagnostics for a scored generation run."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from novelty_distill.evaluation.score_analysis import (
    render_score_summary_markdown,
    summarize_score_payloads,
)
from novelty_distill.evaluation.score_shards import load_score_shard
from novelty_distill.provenance import repository_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, required=True)
    parser.add_argument("--expected-prompts", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    git_commit = repository_commit(Path(__file__).resolve().parents[1])
    if args.samples_per_prompt <= 0:
        raise ValueError("samples-per-prompt must be positive")
    if args.expected_prompts is not None and args.expected_prompts <= 0:
        raise ValueError("expected-prompts must be positive")
    paths = sorted(args.score_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no score shards found in {args.score_dir}")
    if args.expected_prompts is not None and len(paths) != args.expected_prompts:
        raise ValueError(
            f"expected {args.expected_prompts} score shards in {args.score_dir}, found {len(paths)}"
        )
    payloads = tuple(
        load_score_shard(path, samples_per_prompt=args.samples_per_prompt) for path in paths
    )
    summary = summarize_score_payloads(payloads)
    summary["schema_version"] = 1
    summary["git_commit"] = git_commit
    summary["input"] = {
        "path": str(args.score_dir),
        "tree_sha256": _tree_hash(paths),
    }
    _atomic_text(
        args.output,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(args.markdown, render_score_summary_markdown(summary))
    print(
        json.dumps(
            {
                "num_prompts": summary["num_prompts"],
                "num_samples": summary["num_samples"],
                "output": str(args.output),
                "markdown": str(args.markdown),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
