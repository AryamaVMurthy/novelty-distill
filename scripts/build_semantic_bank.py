#!/usr/bin/env python3
"""Export cached clustered embeddings for coverage-residual selection."""

import argparse
from pathlib import Path

from novelty_distill.evaluation.semantic_bank import write_semantic_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("teacher", "student"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_semantic_bank(
        clustered_path=args.clustered,
        output_path=args.output,
        annotation_config=args.annotation_config,
        embedding_cache_dir=args.embedding_cache_dir,
        role=args.role,
    )
    print(f"records={count} output={args.output}")


if __name__ == "__main__":
    main()
