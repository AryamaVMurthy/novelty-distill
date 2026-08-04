import json
import subprocess
import sys
from pathlib import Path

from novelty_distill.evaluation.semantic_calibration import (
    semantic_calibration_protocol_hash,
)

ROOT = Path(__file__).parents[1]


def test_builds_source_blinded_rater_with_safe_embedded_packet(tmp_path: Path) -> None:
    packet = tmp_path / "public.json"
    packet.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_hash": semantic_calibration_protocol_hash(),
                "labels": ["equivalent", "not_equivalent", "uncertain"],
                "entries": [
                    {
                        "blind_id": "sem-1234567890abcdef",
                        "task": "Task with </script><script>alert('x')</script>",
                        "answer_a": "First scientific mechanism",
                        "answer_b": "Second scientific mechanism",
                        "instruction": "Judge central scientific equivalence.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "rater-one.html"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_semantic_equivalence_rater.py"),
            "--packet",
            str(packet),
            "--output",
            str(output),
            "--rater-id",
            "rater-one",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    html = output.read_text(encoding="utf-8")
    assert "Semantic equivalence calibration — rater-one" in html
    assert "sem-1234567890abcdef" in html
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert" in html
    assert "localStorage" in html
    assert "application/jsonl" in html
    assert "confidence: Number(confidence)" in html
    assert "similarity_bin" not in html
    assert "source_pair_sha256" not in html
