#!/usr/bin/env python3
"""Render and validate one promoted TOMATO scale training configuration."""

import argparse
from pathlib import Path

import yaml

from novelty_distill.training.distillm import DistiLLMRunSpec
from novelty_distill.training.gem import GEMRunSpec
from novelty_distill.training.opsd import OPSDRunSpec
from novelty_distill.training.scale_config import render_scale_training_config
from novelty_distill.training.trl import TRLRunSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("trl", "opsd", "gem", "distillm"), required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    rendered = render_scale_training_config(
        base,
        backend=args.backend,
        baseline_id=args.baseline_id,
        train_size=args.train_size,
        seed=args.seed,
    )
    validators = {
        "trl": TRLRunSpec,
        "opsd": OPSDRunSpec,
        "gem": GEMRunSpec,
        "distillm": DistiLLMRunSpec,
    }
    validators[args.backend].model_validate(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
