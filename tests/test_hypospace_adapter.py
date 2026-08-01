import sys
from pathlib import Path

import pytest

from novelty_distill.evaluation.hypospace_adapter import run_official_hypospace

FAKE_BENCHMARK = """
import sys

CAPTURED = {}


class OpenRouterLLM:
    def __init__(self, model, api_key, temperature, base_url):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.base_url = base_url


def setup_llm(llm_type, **kwargs):
    raise AssertionError("the upstream provider factory must be replaced")


def main():
    llm = setup_llm(
        "openrouter", model="novelty-model", api_key="local", temperature=0.7
    )
    CAPTURED.update(base_url=llm.base_url, argv=sys.argv.copy())
"""


def test_runs_official_entrypoint_with_local_sglang_provider(tmp_path: Path) -> None:
    domain_dir = tmp_path / "causal"
    domain_dir.mkdir()
    (domain_dir / "run_causal_benchmark.py").write_text(FAKE_BENCHMARK)

    module = run_official_hypospace(
        repository=tmp_path,
        domain="causal",
        base_url="http://127.0.0.1:30000/v1",
        official_args=["--dataset", "smoke.json"],
    )

    assert module.CAPTURED == {
        "base_url": "http://127.0.0.1:30000/v1",
        "argv": [str(domain_dir / "run_causal_benchmark.py"), "--dataset", "smoke.json"],
    }


def test_rejects_unknown_domain(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported HypoSpace domain"):
        run_official_hypospace(tmp_path, "unknown", "http://localhost/v1", [])


def test_restores_process_arguments_after_upstream_main(tmp_path: Path) -> None:
    domain_dir = tmp_path / "boolean"
    domain_dir.mkdir()
    (domain_dir / "boolean_benchmark.py").write_text(FAKE_BENCHMARK)
    original = sys.argv.copy()

    run_official_hypospace(tmp_path, "boolean", "http://localhost/v1", ["--quiet"])

    assert sys.argv == original
