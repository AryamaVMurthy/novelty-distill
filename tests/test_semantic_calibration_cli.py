import json
import subprocess
import sys
from pathlib import Path

from novelty_distill.evaluation.semantic_calibration import (
    semantic_calibration_protocol_hash,
)

ROOT = Path(__file__).parents[1]


def test_cli_writes_failed_audit_but_exits_nonzero_when_agreement_gate_fails(
    tmp_path: Path,
) -> None:
    protocol_hash = semantic_calibration_protocol_hash()
    blind_ids = ("sem-1111111111111111", "sem-2222222222222222")
    private = tmp_path / "private.json"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    adjudication = tmp_path / "adjudication.jsonl"
    output = tmp_path / "analysis.json"
    private.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_hash": protocol_hash,
                "entries": [
                    {
                        "blind_id": blind_id,
                        "prompt_id": f"prompt-{index}",
                        "left_index": 0,
                        "right_index": 1,
                        "source_pair_sha256": str(index + 1) * 64,
                        "similarity": 0.85 + index * 0.10,
                        "similarity_bin": index,
                        "repeat_of": None,
                    }
                    for index, blind_id in enumerate(blind_ids)
                ],
            }
        ),
        encoding="utf-8",
    )
    first_labels = ("not_equivalent", "equivalent")
    second_labels = ("equivalent", "not_equivalent")
    first.write_text(
        "\n".join(
            json.dumps({"blind_id": blind_id, "label": label, "confidence": 3})
            for blind_id, label in zip(blind_ids, first_labels, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(
            json.dumps({"blind_id": blind_id, "label": label, "confidence": 3})
            for blind_id, label in zip(blind_ids, second_labels, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    adjudication.write_text(
        "\n".join(
            json.dumps({"blind_id": blind_id, "label": label})
            for blind_id, label in zip(blind_ids, first_labels, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_semantic_equivalence_calibration.py"),
            "--private-key",
            str(private),
            "--rater-one",
            str(first),
            "--rater-two",
            str(second),
            "--adjudication",
            str(adjudication),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["status"] == "failed_inter_rater_gate"
    assert audit["selected_threshold"] is None
    assert audit["candidate_threshold"] == 0.95
