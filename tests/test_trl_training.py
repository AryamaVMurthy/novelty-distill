from pathlib import Path

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.trl import (
    build_trl_rows,
    load_canonical_examples,
    load_trl_run_spec,
)


def test_human_sft_uses_canonical_human_target_as_completion() -> None:
    registry = load_baseline_registry(Path("configs/baselines.yaml"))
    baseline = next(item for item in registry.baselines if item.id == "B1")
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Can a coating improve stability?",
            "background_survey": "Humidity damages the catalyst.",
            "fine_grained_hypothesis": "A hydrophobic coating will help.",
            "inspiration": [],
        },
        split="train",
        task="open",
    )

    rows = build_trl_rows(baseline, (example,), teacher_targets={})

    assert len(rows) == 1
    assert rows[0]["completion"] == [
        {"role": "assistant", "content": "A hydrophobic coating will help."}
    ]


def test_diverse_teacher_sft_expands_one_prompt_to_four_saved_targets() -> None:
    registry = load_baseline_registry(Path("configs/baselines.yaml"))
    baseline = next(item for item in registry.baselines if item.id == "B3")
    example = prepare_tomato_record(
        {
            "source_id": "paper-1",
            "research_question": "Can a coating improve stability?",
            "background_survey": "Humidity damages the catalyst.",
            "fine_grained_hypothesis": "A hydrophobic coating will help.",
            "inspiration": [],
        },
        split="train",
        task="open",
    )
    targets = tuple(f"teacher direction {index}" for index in range(4))

    rows = build_trl_rows(
        baseline,
        (example,),
        teacher_targets={"paper-1": {"diverse4": targets}},
    )

    assert [row["completion"][0]["content"] for row in rows] == list(targets)


def test_smoke_run_is_pinned_and_bounded() -> None:
    spec = load_trl_run_spec(Path("configs/training/sft_smoke.yaml"))

    assert spec.baseline_id == "B1"
    assert spec.model == "Qwen/Qwen3-1.7B"
    assert spec.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert spec.max_examples == 1
    assert spec.max_steps == 1
    assert spec.use_peft is True


def test_canonical_loader_honors_smoke_limit(tmp_path: Path) -> None:
    records = [
        prepare_tomato_record(
            {
                "source_id": f"paper-{index}",
                "research_question": "Can a coating improve stability?",
                "background_survey": "Humidity damages the catalyst.",
                "fine_grained_hypothesis": f"Hypothesis {index}.",
                "inspiration": [],
            },
            split="train",
            task="open",
        )
        for index in range(2)
    ]
    path = tmp_path / "canonical.jsonl"
    path.write_text("".join(f"{record.model_dump_json()}\n" for record in records))

    loaded = load_canonical_examples(path, limit=1)

    assert loaded == (records[0],)


def test_on_policy_gkd_keeps_only_ordinary_prompt_in_seed_row() -> None:
    registry = load_baseline_registry(Path("configs/baselines.yaml"))
    baseline = next(item for item in registry.baselines if item.id == "D1")
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

    rows = build_trl_rows(baseline, (example,), teacher_targets={})

    assert rows[0]["messages"][0]["content"] == example.student_prompt
    assert "privileged direction" not in rows[0]["messages"][0]["content"].lower()
    assert rows[0]["messages"][1] == {"role": "assistant", "content": ""}
    assert example.human_target not in str(rows[0])


def test_gkd_smoke_run_pins_both_shared_tokenizer_models() -> None:
    spec = load_trl_run_spec(Path("configs/training/gkd_smoke.yaml"))

    assert spec.baseline_id == "D1"
    assert spec.model == "Qwen/Qwen3-1.7B"
    assert spec.teacher_model == "Qwen/Qwen3-8B"
    assert spec.teacher_revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert spec.max_new_tokens == 64
    assert spec.temperature == 0.8
