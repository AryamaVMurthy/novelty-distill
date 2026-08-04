import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from novelty_distill.evaluation.independent_judge import (
    build_blinded_calibration_sample,
    independent_judge_protocol_hash,
)

ROOT = Path(__file__).parents[1]


def _candidate(prompt_index: int, sample_index: int) -> dict:
    score = 1 + (prompt_index + sample_index) % 5
    return {
        "prompt_id": f"p-{prompt_index}",
        "sample_index": sample_index,
        "text": f"Candidate answer for prompt {prompt_index}, sample {sample_index}.",
        "dimensions": {
            "relevance": 4,
            "feasibility": score,
            "soundness": score,
            "clarity": 4,
            "instruction_compliance": 5,
        },
    }


def test_deepinfra_runner_is_blinded_resumable_and_analyzable(tmp_path: Path) -> None:
    prompts = {f"p-{index}": f"Research task {index}" for index in range(8)}
    candidates = {
        method: [_candidate(prompt_index, 0) for prompt_index in range(8)]
        for method in ("A0", "C1-best1")
    }
    entries = build_blinded_calibration_sample(
        candidates_by_method=candidates,
        prompts=prompts,
        samples_per_method=4,
        repeat_fraction=0.25,
        seed=17,
    )
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_hash": independent_judge_protocol_hash(),
                "entries": [entry.model_dump(mode="json") for entry in entries],
            }
        ),
        encoding="utf-8",
    )

    received: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            assert self.headers["Authorization"] == "Bearer test-only-key"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                received.append(body)
                request_index = len(received)
            content = {
                "relevance": 4,
                "feasibility": 3,
                "soundness": 4,
                "clarity": 4,
                "instruction_compliance": 5,
                "fatal_flaw": False,
                "brief_rationale": "The mechanism is testable but one control is vague.",
            }
            response = json.dumps(
                {
                    "id": f"mock-request-{request_index}",
                    "model": "mock-independent-model-revision",
                    "choices": [{"message": {"content": json.dumps(content)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    responses = tmp_path / "responses"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_deepinfra_judge_calibration.py"),
        "--packet",
        str(packet),
        "--output-dir",
        str(responses),
        "--model",
        "mock/model",
        "--endpoint",
        endpoint,
        "--concurrency",
        "3",
    ]
    environment = {**os.environ, "DEEPINFRA_API_KEY": "test-only-key"}
    try:
        first = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert len(received) == len(entries)
        assert '"created": ' + str(len(entries)) in first.stderr
        for body in received:
            serialized = json.dumps(body)
            assert "A0" not in serialized
            assert "C1-best1" not in serialized
            assert "qwen_dimensions" not in serialized
            assert "repeat_of" not in serialized

        second = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert len(received) == len(entries)
        assert '"reused": ' + str(len(entries)) in second.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    output = tmp_path / "analysis.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_independent_judge_calibration.py"),
            "--packet",
            str(packet),
            "--responses",
            str(responses),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["counts"] == {"originals": 8, "repeats": 2, "total": 10}
    assert summary["repeat_reliability"]["feasibility"]["exact_rate"] == 1
    assert summary["provenance"]["provider_models"] == ["mock-independent-model-revision"]
    assert set(summary["paired_contrasts"]) == {"C1-best1_minus_A0"}
