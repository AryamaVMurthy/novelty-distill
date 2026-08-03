from pathlib import Path

from novelty_distill.config import load_baseline_registry
from novelty_distill.data.tomato import prepare_tomato_record
from novelty_distill.training.trl import (
    build_trl_rows,
    configure_gkd_generation,
    encode_prompt_preserving_chatml_example,
    load_canonical_examples,
    load_trl_run_spec,
    override_trl_baseline,
)


def test_gkd_generation_uses_frozen_teacher_sampling_controls() -> None:
    class GenerationConfig:
        temperature = 99.0
        top_p = 0.95
        top_k = 0
        min_p = None

    class Trainer:
        generation_config = GenerationConfig()

    spec = load_trl_run_spec(Path("configs/training/gkd_tomato1k.yaml"))

    controls = configure_gkd_generation(Trainer(), spec)

    assert controls == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": 512,
        "student_thinking": False,
    }
    assert Trainer.generation_config.temperature == 0.7
    assert Trainer.generation_config.top_p == 0.8
    assert Trainer.generation_config.top_k == 20
    assert Trainer.generation_config.min_p == 0.0


def test_gkd_collator_preserves_prompt_and_truncates_completion_tail() -> None:
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            assert kwargs["enable_thinking"] is False
            return "PP" if len(messages) == 1 else "PPabcdefgh"

        def __call__(self, text, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": [ord(character) for character in text]}

    encoded = encode_prompt_preserving_chatml_example(
        {
            "messages": [
                {"role": "user", "content": "problem"},
                {"role": "assistant", "content": "long completion"},
            ]
        },
        tokenizer=CharacterTokenizer(),
        max_length=6,
    )

    assert encoded["prompt_ids"] == [ord("P"), ord("P")]
    assert encoded["input_ids"] == [ord(char) for char in "PPabcd"]
    assert encoded["labels"] == [-100, -100, *[ord(char) for char in "abcd"]]
    assert encoded["truncated_completion_tokens"] == 4


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


def test_random4_rows_derive_from_canonical_all8_without_mutating_artifact() -> None:
    registry = load_baseline_registry(Path("configs/exposure_sensitivity_baselines.yaml"))
    baseline = next(item for item in registry.baselines if item.id == "F2-random4")
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
    all8 = tuple(f"teacher-{index}" for index in range(8))

    rows = build_trl_rows(
        baseline,
        (example,),
        teacher_targets={example.id: {"all8": all8}},
        selection_seed=17,
    )

    selected = {row["completion"][0]["content"] for row in rows}
    assert len(rows) == 4
    assert len(selected) == 4
    assert selected < set(all8)


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
    # TRL's official ChatML collator drops the prompt when a completion alone
    # reaches max_length; the first canonical human target is 804 tokens.
    assert spec.max_length == 1024
    assert spec.max_new_tokens == 64
    assert spec.temperature == 0.8
    assert spec.top_p == 0.8
    assert spec.top_k == 20
    assert spec.min_p == 0.0
    assert spec.student_thinking is False


def test_teacher_seqkd_config_requires_the_versioned_target_artifact() -> None:
    spec = override_trl_baseline(
        load_trl_run_spec(Path("configs/training/sft_smoke.yaml")), "B2b"
    )

    assert spec.baseline_id == "B2b"
    assert spec.teacher_targets == Path("data/teacher-targets.json")
    assert spec.output_dir == Path("checkpoints/B2b-smoke")


def test_baseline_override_can_name_a_non_smoke_run() -> None:
    spec = override_trl_baseline(
        load_trl_run_spec(Path("configs/training/sft_smoke.yaml")),
        "B2b",
        run_suffix="tomato1k-seed17",
    )

    assert spec.output_dir == Path("checkpoints/B2b-tomato1k-seed17")


def test_research_run_checkpoints_before_the_six_hour_boundary() -> None:
    sft = load_trl_run_spec(Path("configs/training/sft_tomato1k.yaml"))
    gkd = load_trl_run_spec(Path("configs/training/gkd_tomato1k.yaml"))

    assert sft.save_steps == 25
    assert gkd.save_steps == 25
    assert (gkd.temperature, gkd.top_p, gkd.top_k, gkd.min_p) == (0.7, 0.8, 20, 0.0)
