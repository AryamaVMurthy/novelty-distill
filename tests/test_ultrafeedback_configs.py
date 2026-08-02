from pathlib import Path

from novelty_distill.training.trl import load_trl_run_spec

ROOT = Path(__file__).resolve().parents[1]


def test_ultrafeedback_pilot_configs_share_data_and_effective_exposure() -> None:
    sft = load_trl_run_spec(ROOT / "configs" / "training" / "sft_ultrafeedback1k.yaml")
    gkd = load_trl_run_spec(ROOT / "configs" / "training" / "gkd_ultrafeedback1k.yaml")

    assert sft.input == gkd.input == Path("data/ultrafeedback-train-1000.jsonl")
    assert (
        sft.teacher_targets == gkd.teacher_targets == Path("data/ultrafeedback-targets-1000.json")
    )
    assert sft.max_examples == gkd.max_examples == 1000
    assert sft.max_steps * sft.gradient_accumulation_steps == 1000
    assert gkd.max_steps * gkd.gradient_accumulation_steps == 1000
    assert sft.model == gkd.model == "Qwen/Qwen3-1.7B"
    assert gkd.teacher_model == "Qwen/Qwen3-8B"
