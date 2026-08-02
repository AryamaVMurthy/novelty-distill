from pathlib import Path

from novelty_distill.training.distillm import load_distillm_run_spec
from novelty_distill.training.gem import load_gem_run_spec
from novelty_distill.training.opsd import load_opsd_run_spec
from novelty_distill.training.trl import load_trl_run_spec

STUDENT = "Qwen/Qwen3-4B"
STUDENT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TEACHER = "Qwen/Qwen3-14B"
TEACHER_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"


def test_tomato1k_configs_share_main_models_data_size_and_optimizer_budget() -> None:
    sft = load_trl_run_spec(Path("configs/training/sft_tomato1k.yaml"))
    gkd = load_trl_run_spec(Path("configs/training/gkd_tomato1k.yaml"))
    opsd = load_opsd_run_spec(Path("configs/training/opsd_tomato1k.yaml"))
    gem = load_gem_run_spec(Path("configs/training/gem_tomato1k.yaml"))
    distillm = load_distillm_run_spec(Path("configs/training/distillm_tomato1k.yaml"))

    assert {sft.model, gkd.model, opsd.model, gem.model, distillm.student_model} == {
        STUDENT
    }
    assert {
        sft.revision,
        gkd.revision,
        opsd.revision,
        gem.revision,
        distillm.student_revision,
    } == {STUDENT_REVISION}
    assert gkd.teacher_model == distillm.teacher_model == TEACHER
    assert gkd.teacher_revision == distillm.teacher_revision == TEACHER_REVISION
    assert {sft.max_examples, gkd.max_examples, opsd.max_examples, gem.max_examples} == {
        1000
    }
    assert distillm.max_examples == 1000
    effective_examples = {
        sft.max_steps * sft.per_device_train_batch_size * sft.gradient_accumulation_steps,
        gkd.max_steps * gkd.per_device_train_batch_size * gkd.gradient_accumulation_steps,
        opsd.max_steps
        * opsd.per_device_train_batch_size
        * opsd.gradient_accumulation_steps,
        gem.max_steps
        * gem.per_device_train_batch_size
        * gem.gradient_accumulation_steps,
        distillm.max_steps * distillm.batch_size,
    }
    assert effective_examples == {1000}
    assert {sft.seed, gkd.seed, opsd.seed, gem.seed, distillm.seed} == {17}


def test_main_model_smokes_exercise_the_memory_heavy_backends() -> None:
    gkd = load_trl_run_spec(Path("configs/training/gkd_main_model_smoke.yaml"))
    opsd = load_opsd_run_spec(Path("configs/training/opsd_main_model_smoke.yaml"))
    distillm = load_distillm_run_spec(
        Path("configs/training/distillm_main_model_smoke.yaml")
    )

    assert gkd.model == opsd.model == distillm.student_model == STUDENT
    assert gkd.teacher_model == distillm.teacher_model == TEACHER
    assert gkd.max_new_tokens == opsd.max_completion_length == 512
    assert gkd.max_steps == opsd.max_steps == distillm.max_steps == 1
