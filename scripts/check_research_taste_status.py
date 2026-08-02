#!/usr/bin/env python3
"""Report whether all generations have current research-taste annotations."""

import argparse
import json
from pathlib import Path

import yaml

from novelty_distill.evaluation.research_taste import ResearchTasteSpec
from novelty_distill.evaluation.taste_shards import research_taste_run_status
from novelty_distill.generation.sglang import GenerationSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--taste-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_spec = GenerationSpec.model_validate(
        yaml.safe_load(args.generation_config.read_text(encoding="utf-8"))
    )
    config = yaml.safe_load(args.taste_config.read_text(encoding="utf-8"))
    annotator = ResearchTasteSpec(
        model=config["annotator_model"],
        revision=config["annotator_revision"],
        max_tokens=config["annotator_max_tokens"],
    )
    total, pending = research_taste_run_status(
        generation_dir=args.generation_dir,
        output_dir=args.output_dir,
        generation_spec=generation_spec,
        annotator=annotator,
    )
    print(json.dumps({"complete": pending == 0, "pending": pending, "total": total}))
    if pending:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
