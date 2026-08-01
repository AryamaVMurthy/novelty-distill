"""Thin adapter from HypoSpace's official OpenRouter client to local SGLang."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

DOMAIN_ENTRYPOINTS = {
    "causal": "run_causal_benchmark.py",
    "3d": "run_3d_benchmark.py",
    "boolean": "boolean_benchmark.py",
}


def official_artifact_args(
    domain: str,
    checkpoint_dir: Path,
    output: Path,
) -> list[str]:
    """Return artifact flags supported by the pinned official domain CLI."""
    if domain not in DOMAIN_ENTRYPOINTS:
        supported = ", ".join(DOMAIN_ENTRYPOINTS)
        raise ValueError(
            f"unsupported HypoSpace domain {domain!r}; expected one of {supported}"
        )
    if domain == "boolean":
        return []
    return [
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output",
        str(output),
    ]


def _load_entrypoint(repository: Path, domain: str) -> tuple[ModuleType, Path]:
    try:
        filename = DOMAIN_ENTRYPOINTS[domain]
    except KeyError as error:
        supported = ", ".join(DOMAIN_ENTRYPOINTS)
        raise ValueError(
            f"unsupported HypoSpace domain {domain!r}; expected one of {supported}"
        ) from error

    entrypoint = repository.resolve() / domain / filename
    if not entrypoint.is_file():
        raise FileNotFoundError(f"official HypoSpace entrypoint not found: {entrypoint}")

    spec = importlib.util.spec_from_file_location(f"_hypospace_{domain}_benchmark", entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load official HypoSpace entrypoint: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, entrypoint


def _local_sglang_factory(module: ModuleType, base_url: str):
    def setup_llm(llm_type: str, **kwargs: Any):
        if llm_type != "openrouter":
            raise ValueError("local HypoSpace evaluation requires llm.type=openrouter")
        return module.OpenRouterLLM(
            model=kwargs.get("model", "novelty-model"),
            api_key=kwargs.get("api_key", "local"),
            temperature=kwargs.get("temperature", 0.7),
            base_url=base_url.rstrip("/"),
        )

    return setup_llm


def run_official_hypospace(
    repository: Path,
    domain: str,
    base_url: str,
    official_args: list[str],
) -> ModuleType:
    """Run an official HypoSpace CLI while routing its existing client to SGLang."""
    domain_dir = repository.resolve() / domain
    original_argv = sys.argv.copy()
    sys.path.insert(0, str(domain_dir))
    try:
        module, entrypoint = _load_entrypoint(repository, domain)
        module.setup_llm = _local_sglang_factory(module, base_url)
        sys.argv = [str(entrypoint), *official_args]
        module.main()
        return module
    finally:
        sys.argv = original_argv
        sys.path.remove(str(domain_dir))
