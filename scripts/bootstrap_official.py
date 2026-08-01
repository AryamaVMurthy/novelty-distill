#!/usr/bin/env python3
"""Fetch only explicitly requested official repositories at their pinned commits."""

import argparse
from pathlib import Path

from novelty_distill.official import checkout_official_repository, load_official_repositories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("third_party/manifest.yaml"))
    args = parser.parse_args()

    repositories = load_official_repositories(args.manifest)
    unknown = sorted(set(args.names) - repositories.keys())
    if unknown:
        raise ValueError(f"unknown official repositories: {', '.join(unknown)}")
    for name in args.names:
        checkout = checkout_official_repository(repositories[name], args.destination)
        print(f"{name}\t{repositories[name].commit}\t{checkout}")


if __name__ == "__main__":
    main()
