import sys
from pathlib import Path

import pytest

from novelty_distill.evaluation.hypospace_adapter import (
    official_artifact_args,
    run_official_hypospace,
    validate_hypospace_result,
)

FAKE_BENCHMARK = """
import sys

CAPTURED = {}


class FakeRequests:
    def __init__(self):
        self.payload = None

    def post(self, url, **kwargs):
        self.payload = kwargs["json"]


requests = FakeRequests()


class OpenRouterLLM:
    def __init__(self, model, api_key, temperature, max_tokens, base_url):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    def query_with_usage(self, prompt):
        requests.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": [{"content": prompt}]},
        )


def setup_llm(llm_type, **kwargs):
    raise AssertionError("the upstream provider factory must be replaced")


def main():
    llm = setup_llm(
        "openrouter", model="novelty-model", api_key="local", temperature=0.7
    )
    llm.query_with_usage("test prompt")
    CAPTURED.update(
        base_url=llm.base_url,
        max_tokens=llm.max_tokens,
        request_payload=requests.payload,
        argv=sys.argv.copy(),
    )
"""


def test_runs_official_entrypoint_with_local_sglang_provider(tmp_path: Path) -> None:
    domain_dir = tmp_path / "causal"
    domain_dir.mkdir()
    (domain_dir / "run_causal_benchmark.py").write_text(FAKE_BENCHMARK)

    module = run_official_hypospace(
        repository=tmp_path,
        domain="causal",
        base_url="http://127.0.0.1:30000/v1",
        max_tokens=512,
        official_args=["--dataset", "smoke.json"],
    )

    assert module.CAPTURED == {
        "base_url": "http://127.0.0.1:30000/v1",
        "max_tokens": 512,
        "request_payload": {
            "model": "novelty-model",
            "messages": [{"content": "test prompt"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "argv": [str(domain_dir / "run_causal_benchmark.py"), "--dataset", "smoke.json"],
    }


def test_rejects_unknown_domain(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported HypoSpace domain"):
        run_official_hypospace(tmp_path, "unknown", "http://localhost/v1", 512, [])


def test_official_artifact_args_match_domain_clis(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    output = tmp_path / "result.json"

    expected = [
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output",
        str(output),
    ]
    assert official_artifact_args("causal", checkpoint_dir, output) == expected
    assert official_artifact_args("3d", checkpoint_dir, output) == expected
    assert official_artifact_args("boolean", checkpoint_dir, output) == []


def test_restores_process_arguments_after_upstream_main(tmp_path: Path) -> None:
    domain_dir = tmp_path / "boolean"
    domain_dir.mkdir()
    (domain_dir / "boolean_benchmark.py").write_text(FAKE_BENCHMARK)
    original = sys.argv.copy()

    run_official_hypospace(tmp_path, "boolean", "http://localhost/v1", 512, ["--quiet"])

    assert sys.argv == original


def test_rejects_official_result_with_request_errors(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(
        '{"error_summary":{"total_errors":2,"error_rate":1.0,'
        '"error_types":{"http_error_400":6}}}'
    )

    with pytest.raises(RuntimeError, match="2 failed samples.*http_error_400"):
        validate_hypospace_result(result)


def test_accepts_official_result_without_request_errors(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(
        '{"error_summary":{"total_errors":0,"error_rate":0.0,"error_types":{}}}'
    )

    validate_hypospace_result(result)
