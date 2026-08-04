#!/usr/bin/env python3
"""Write a content-bound integrity audit for canonical TOMATO historical targets."""

import argparse
from pathlib import Path

from novelty_distill.data.target_integrity import audit_target_integrity
from novelty_distill.data.tomato import CanonicalExample
from novelty_distill.training.provenance import atomic_json, file_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngram-size", type=int, default=12)
    parser.add_argument("--minimum-document-frequency", type=int, default=3)
    parser.add_argument("--maximum-prompt-token-coverage", type=float, default=0.25)
    parser.add_argument("--word-limit", type=int, default=300)
    parser.add_argument("--maximum-candidates", type=int, default=100)
    args = parser.parse_args()

    rows = []
    for path in args.input:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(CanonicalExample.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(f"invalid canonical row at {path}:{line_number}") from error

    payload = {
        "schema_version": 1,
        "inputs": [file_provenance(path) for path in args.input],
        "integrity": audit_target_integrity(
            rows,
            ngram_size=args.ngram_size,
            minimum_document_frequency=args.minimum_document_frequency,
            maximum_prompt_token_coverage=args.maximum_prompt_token_coverage,
            word_limit=args.word_limit,
            maximum_candidates=args.maximum_candidates,
        ),
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
