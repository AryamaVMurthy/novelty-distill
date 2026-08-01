#!/usr/bin/env python3
"""Run pinned official HypoSpace with its OpenRouter client pointed at SGLang."""

import argparse
from pathlib import Path

from novelty_distill.evaluation.hypospace_adapter import (
    official_artifact_args,
    run_official_hypospace,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--domain", choices=("causal", "3d", "boolean"), required=True)
    parser.add_argument("--base-url", required=True)
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
        official_args=official_args,
    )


if __name__ == "__main__":
    main()
