"""Launch pinned official GEM with its narrow Transformers compatibility shim."""

import argparse
import importlib
import runpy
import sys
from pathlib import Path

from packaging.version import Version
from transformers import __version__ as transformers_version

from novelty_distill.training.gem import install_gem_trainer_compat


def run_official_gem(train_path: Path, official_args: list[str]) -> None:
    """Execute official GEM after widening one upstream callback signature."""

    train_path = train_path.resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"official GEM entrypoint not found: {train_path}")
    official_dir = str(train_path.parent)
    trainer_module_name = (
        "sft_trainer_v2"
        if Version(transformers_version) >= Version("4.46.0")
        else "sft_trainer"
    )
    original_argv = sys.argv.copy()
    sys.path.insert(0, official_dir)
    try:
        trainer_module = importlib.import_module(trainer_module_name)
        install_gem_trainer_compat(trainer_module.SFTTrainer)
        sys.argv = [str(train_path), *official_args]
        runpy.run_path(str(train_path), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path.remove(official_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-train", type=Path, required=True)
    args, official_args = parser.parse_known_args()
    run_official_gem(args.official_train, official_args)


if __name__ == "__main__":
    main()
