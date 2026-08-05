import json
import subprocess
import sys
from pathlib import Path

from novelty_distill.evaluation.independent_judge import (
    build_blinded_calibration_sample,
    independent_judge_protocol_hash,
)

ROOT = Path(__file__).parents[1]


def test_migrate_independent_judge_packet_preserves_frozen_selection(tmp_path: Path) -> None:
    entries = build_blinded_calibration_sample(
        candidates_by_method={
            "A0": [
                {
                    "prompt_id": f"p-{index}",
                    "sample_index": 0,
                    "text": f"answer {index}",
                    "dimensions": {
                        "relevance": 4,
                        "feasibility": 3,
                        "soundness": 4,
                        "clarity": 4,
                        "instruction_compliance": 5,
                    },
                }
                for index in range(4)
            ]
        },
        prompts={f"p-{index}": f"prompt {index}" for index in range(4)},
        samples_per_method=4,
        repeat_fraction=0,
        seed=17,
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_hash": "a" * 64,
                "sampling": {"design": "frozen-v3", "seed": 17},
                "sources": {"prompt_sha256": "b" * 64},
                "entries": [entry.model_dump(mode="json") for entry in entries],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/migrate_independent_judge_packet.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--expected-old-protocol-hash",
            "a" * 64,
            "--reason",
            "provider JSON-schema transport ignored constraints",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    migrated = json.loads(output.read_text(encoding="utf-8"))
    assert migrated["protocol_hash"] == independent_judge_protocol_hash()
    assert migrated["entries"] == json.loads(source.read_text())["entries"]
    assert migrated["sampling"] == {"design": "frozen-v3", "seed": 17}
    assert migrated["protocol_migration"]["old_protocol_hash"] == "a" * 64
    assert migrated["protocol_migration"]["reason"] == (
        "provider JSON-schema transport ignored constraints"
    )
    assert len(migrated["protocol_migration"]["source_packet_sha256"]) == 64
