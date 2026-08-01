from pathlib import Path

from novelty_distill.config import load_baseline_registry
from novelty_distill.integrations import build_backend_plan


def test_every_training_baseline_maps_to_an_official_entrypoint() -> None:
    registry = load_baseline_registry(Path("configs/baselines.yaml"))

    plans = {
        baseline.id: build_backend_plan(baseline)
        for baseline in registry.baselines
        if baseline.backend != "evaluation"
    }

    assert plans["B1"].repository == "trl"
    assert plans["B1"].entrypoint == "trl.SFTTrainer"
    assert plans["C1-human"].entrypoint == "trl.experimental.gkd.GKDTrainer"
    assert plans["C1-human"].arguments["lmbda"] == 0.0
    assert plans["C1-human"].arguments["beta"] == 0.0
    assert plans["D2"].arguments["lmbda"] == 1.0
    assert plans["D2"].arguments["beta"] == 1.0
    assert plans["E2"].entrypoint == "opsd_train.py"
    assert plans["B4"].entrypoint == "train.py"
    assert plans["C3"].entrypoint == "finetune.py"
    assert all(plan.repository and plan.entrypoint for plan in plans.values())
