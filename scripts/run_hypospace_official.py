#!/usr/bin/env python3
"""Run pinned official HypoSpace with its OpenRouter client pointed at SGLang."""

import argparse
from pathlib import Path

from novelty_distill.evaluation.hypospace_adapter import (
    official_artifact_args,
    run_official_hypospace,
    validate_hypospace_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--domain", choices=("causal", "3d", "boolean"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("official_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    official_args = args.official_args
    if official_args[:1] == ["--"]:
        official_args = official_args[1:]
    official_args.extend(
        official_artifact_args(args.domain, args.checkpoint_dir, args.output)
    )
    run_official_hypospace(
        repository=args.repository,
        domain=args.domain,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        official_args=official_args,
    )
    result_path = args.output
    if args.domain == "boolean":
        candidates = list(Path.cwd().glob("results/*.json"))
        if not candidates:
            raise FileNotFoundError("official Boolean HypoSpace result was not created")
        result_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    validate_hypospace_result(result_path)


if __name__ == "__main__":
    main()
