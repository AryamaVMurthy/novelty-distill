#!/usr/bin/env python3
"""Report whether all generation shards have current quality-score shards."""

import argparse
import json
from pathlib import Path

import yaml

from novelty_distill.evaluation.score_shards import score_run_status
from novelty_distill.evaluation.teacher_annotation import JudgeSpec
from novelty_distill.generation.sglang import GenerationSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_spec = GenerationSpec.model_validate(
        yaml.safe_load(args.generation_config.read_text(encoding="utf-8"))
    )
    annotation = yaml.safe_load(args.annotation_config.read_text(encoding="utf-8"))
    judge = JudgeSpec(model=annotation["judge_model"], revision=annotation["judge_revision"])
    total, pending = score_run_status(
        generation_dir=args.generation_dir,
        output_dir=args.output_dir,
        generation_spec=generation_spec,
        judge=judge,
    )
    print(json.dumps({"complete": pending == 0, "pending": pending, "total": total}))
    if pending:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
