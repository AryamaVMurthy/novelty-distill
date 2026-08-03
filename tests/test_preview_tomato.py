import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_preview_tomato_selects_and_pretty_prints_a_record_by_id(tmp_path: Path) -> None:
    rows = (
        {"id": "p1", "student_prompt": "first", "human_target": "answer one"},
        {"id": "p2", "student_prompt": "second", "human_target": "answer two"},
    )
    dataset = tmp_path / "tomato.jsonl"
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preview_tomato.py"),
            str(dataset),
            "--id",
            "p2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == rows[1]

