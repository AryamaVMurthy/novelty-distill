#!/usr/bin/env python3
"""Print the content identity of a local inference checkpoint or adapter."""

import argparse
from pathlib import Path

from novelty_distill.generation.sglang import model_artifact_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    print(model_artifact_identity(args.path))


if __name__ == "__main__":
    main()
