#!/usr/bin/env python3
"""Merge a complete set of teacher-clustering partitions."""

import argparse
import json
from pathlib import Path

from novelty_distill.evaluation.cluster_shards import merge_cluster_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("part-*.jsonl"))
    metadata = merge_cluster_shards(
        paths,
        output=args.output,
        expected_prompts=args.expected_prompts,
        expected_num_shards=args.num_shards,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
