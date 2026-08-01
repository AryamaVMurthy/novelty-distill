from pathlib import Path

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.opsd import build_opsd_rows, load_opsd_run_spec


def test_privileged_opsd_smoke_is_pinned_and_uses_official_row_schema() -> None:
    spec = load_opsd_run_spec(Path("configs/training/opsd_smoke.yaml"))
    registry = load_baseline_registry(Path("configs/baselines.yaml"))
    baseline = next(item for item in registry.baselines if item.id == spec.baseline_id)
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Can a coating improve stability?",
            "background_survey": "Humidity damages the catalyst.",
            "fine_grained_hypothesis": "A hydrophobic coating will help.",
            "inspiration": [{"insp": "A privileged direction."}],
        },
        split="train",
        task="open",
    )

    rows = build_opsd_rows(baseline, (example,))

    assert spec.model == "Qwen/Qwen3-1.7B"
    assert spec.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert spec.max_steps == 1
    assert rows[0]["problem"] == example.student_prompt
    assert "A hydrophobic coating" in rows[0]["solution"]
    assert "A privileged direction" in rows[0]["solution"]
