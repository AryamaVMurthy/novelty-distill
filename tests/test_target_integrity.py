import json
import subprocess
import sys
from pathlib import Path

from novelty_distill.data.target_integrity import audit_target_integrity
from novelty_distill.data.tomato import prepare_tomato_record


def _row(source_id: str, question: str, target: str):
    return prepare_tomato_record(
        {
            "source_id": source_id,
            "research_question": question,
            "background_survey": "A short domain-specific background.",
            "fine_grained_hypothesis": target,
            "inspiration": [{"insp": "A relevant inspiration."}],
        },
        split="train",
        task="open",
    )


def test_target_integrity_finds_repeated_passage_unsupported_by_prompts() -> None:
    pasted = (
        "Oxytocin signaling through its receptor activates calcium dependent pathways in "
        "adult born granule cells and changes their long term excitability."
    )
    rows = (
        _row("a", "How can mitochondrial damage in Parkinson disease be reduced?", pasted),
        _row("b", "How can brain invasion during viral infection be measured?", pasted),
        _row("c", "Does oxytocin alter adult born granule cells?", pasted),
    )

    report = audit_target_integrity(
        rows,
        ngram_size=8,
        minimum_document_frequency=2,
        maximum_prompt_token_coverage=0.25,
        word_limit=10,
    )

    assert report["rows"] == 3
    assert report["target_equals_privileged_count"] == 3
    assert report["over_word_limit_count"] == 3
    assert report["suspicious_row_ids"] == ["a", "b"]
    assert report["suspicious_repeated_ngrams"][0]["document_frequency"] == 3
    assert report["suspicious_repeated_ngrams"][0]["unsupported_prompt_ids"] == ["a", "b"]


def test_target_integrity_cli_writes_a_content_bound_report(tmp_path: Path) -> None:
    dataset = tmp_path / "tomato.jsonl"
    output = tmp_path / "audit.json"
    rows = (
        _row("a", "Question about mitochondria?", "Repeated foreign mechanism text goes here."),
        _row("b", "Question about viral entry?", "Repeated foreign mechanism text goes here."),
    )
    dataset.write_text("".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_tomato_targets.py",
            "--input",
            str(dataset),
            "--output",
            str(output),
            "--ngram-size",
            "3",
            "--minimum-document-frequency",
            "2",
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["inputs"][0]["path"] == str(dataset)
    assert len(report["inputs"][0]["sha256"]) == 64
    assert report["integrity"]["suspicious_row_ids"] == ["a", "b"]
