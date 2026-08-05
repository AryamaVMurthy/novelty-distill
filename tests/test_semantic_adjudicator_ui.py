import json
import subprocess
import sys
from pathlib import Path

from novelty_distill.evaluation.semantic_calibration import (
    semantic_calibration_protocol_hash,
)

ROOT = Path(__file__).parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_builds_source_blinded_adjudicator_for_flagged_originals_only(
    tmp_path: Path,
) -> None:
    protocol_hash = semantic_calibration_protocol_hash()
    original_agree = "sem-1111111111111111"
    original_disagree = "sem-2222222222222222"
    repeated_disagree = "sem-3333333333333333"
    public_entries = [
        {
            "blind_id": original_agree,
            "task": "Agreement task",
            "answer_a": "Agreement answer A",
            "answer_b": "Agreement answer B",
            "instruction": "Judge central scientific equivalence.",
        },
        {
            "blind_id": original_disagree,
            "task": "Disagreement task with </script><script>alert('x')</script>",
            "answer_a": "Disagreement answer A",
            "answer_b": "Disagreement answer B",
            "instruction": "Judge central scientific equivalence.",
        },
        {
            "blind_id": repeated_disagree,
            "task": "Disagreement task",
            "answer_a": "Disagreement answer B",
            "answer_b": "Disagreement answer A",
            "instruction": "Judge central scientific equivalence.",
        },
    ]
    private_entries = [
        {
            "blind_id": original_agree,
            "prompt_id": "private-prompt-one",
            "left_index": 0,
            "right_index": 1,
            "source_pair_sha256": "a" * 64,
            "similarity": 0.91,
            "similarity_bin": 3,
            "repeat_of": None,
        },
        {
            "blind_id": original_disagree,
            "prompt_id": "private-prompt-two",
            "left_index": 0,
            "right_index": 1,
            "source_pair_sha256": "b" * 64,
            "similarity": 0.95,
            "similarity_bin": 5,
            "repeat_of": None,
        },
        {
            "blind_id": repeated_disagree,
            "prompt_id": "private-prompt-two",
            "left_index": 0,
            "right_index": 1,
            "source_pair_sha256": "b" * 64,
            "similarity": 0.95,
            "similarity_bin": 5,
            "repeat_of": original_disagree,
        },
    ]
    packet = tmp_path / "public.json"
    private = tmp_path / "private.json"
    rater_one = tmp_path / "rater-one.jsonl"
    rater_two = tmp_path / "rater-two.jsonl"
    output = tmp_path / "adjudicator.html"
    _write_json(
        packet,
        {
            "schema_version": 1,
            "protocol_hash": protocol_hash,
            "labels": ["equivalent", "not_equivalent", "uncertain"],
            "entries": public_entries,
        },
    )
    _write_json(
        private,
        {
            "schema_version": 1,
            "protocol_hash": protocol_hash,
            "entries": private_entries,
        },
    )
    rater_one.write_text(
        "\n".join(
            json.dumps({"blind_id": entry["blind_id"], "label": "equivalent", "confidence": 3})
            for entry in public_entries
        )
        + "\n",
        encoding="utf-8",
    )
    rater_two.write_text(
        "\n".join(
            json.dumps(
                {
                    "blind_id": entry["blind_id"],
                    "label": (
                        "equivalent" if entry["blind_id"] == original_agree else "not_equivalent"
                    ),
                    "confidence": 2,
                }
            )
            for entry in public_entries
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_semantic_equivalence_adjudicator.py"),
            "--packet",
            str(packet),
            "--private-key",
            str(private),
            "--rater-one",
            str(rater_one),
            "--rater-two",
            str(rater_two),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    html = output.read_text(encoding="utf-8")
    assert original_disagree in html
    assert "Disagreement answer A" in html
    assert original_agree not in html
    assert repeated_disagree not in html
    assert "private-prompt" not in html
    assert "similarity" not in html
    assert "rater-one" not in html
    assert "rater-two" not in html
    assert "application/jsonl" in html
    assert "adjudication.jsonl" in html
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert" in html


def test_adjudicator_rejects_incomplete_rater_exports_without_writing_output(
    tmp_path: Path,
) -> None:
    protocol_hash = semantic_calibration_protocol_hash()
    blind_ids = ("sem-1111111111111111", "sem-2222222222222222")
    packet = tmp_path / "public.json"
    private = tmp_path / "private.json"
    rater_one = tmp_path / "rater-one.jsonl"
    rater_two = tmp_path / "rater-two.jsonl"
    output = tmp_path / "adjudicator.html"
    _write_json(
        packet,
        {
            "schema_version": 1,
            "protocol_hash": protocol_hash,
            "labels": ["equivalent", "not_equivalent", "uncertain"],
            "entries": [
                {
                    "blind_id": blind_id,
                    "task": f"Task {index}",
                    "answer_a": f"Answer A {index}",
                    "answer_b": f"Answer B {index}",
                    "instruction": "Judge central scientific equivalence.",
                }
                for index, blind_id in enumerate(blind_ids)
            ],
        },
    )
    _write_json(
        private,
        {
            "schema_version": 1,
            "protocol_hash": protocol_hash,
            "entries": [
                {
                    "blind_id": blind_id,
                    "prompt_id": f"private-{index}",
                    "left_index": 0,
                    "right_index": 1,
                    "source_pair_sha256": str(index + 1) * 64,
                    "similarity": 0.90 + index * 0.01,
                    "similarity_bin": index,
                    "repeat_of": None,
                }
                for index, blind_id in enumerate(blind_ids)
            ],
        },
    )
    complete_row = json.dumps({"blind_id": blind_ids[0], "label": "equivalent", "confidence": 3})
    rater_one.write_text(complete_row + "\n", encoding="utf-8")
    rater_two.write_text(complete_row + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_semantic_equivalence_adjudicator.py"),
            "--packet",
            str(packet),
            "--private-key",
            str(private),
            "--rater-one",
            str(rater_one),
            "--rater-two",
            str(rater_two),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "exactly cover every packet blind id" in result.stderr
    assert not output.exists()
