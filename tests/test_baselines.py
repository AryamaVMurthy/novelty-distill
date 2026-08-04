from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from novelty_distill.config import BaselineConfig, load_baseline_registry
from novelty_distill.training.opsd import load_opsd_run_spec, override_opsd_baseline
from novelty_distill.training.trl import load_trl_run_spec, override_trl_baseline


def test_registry_contains_every_planned_baseline_once() -> None:
    registry = load_baseline_registry(Path("configs/baselines.yaml"))

    expected = {
        "A0",
        "A1",
        "A2",
        "A3",
        "B1",
        "B2a",
        "B2b",
        "B2c",
        "B3",
        "B4",
        "C1-human",
        "C1-best1",
        "C1-diverse4",
        "C2-human",
        "C2-best1",
        "C2-diverse4",
        "C3",
        "D1",
        "D2",
        "D3",
        "E1",
        "E2",
        "E3",
        "E4",
    }

    assert {baseline.id for baseline in registry.baselines} == expected
    assert len(registry.baselines) == len(expected)
    assert all(baseline.official_source for baseline in registry.baselines)


def test_registry_marks_the_unsupported_static_opsd_control_fail_closed() -> None:
    registry = load_baseline_registry(Path("configs/baselines.yaml"))
    by_id = {baseline.id: baseline for baseline in registry.baselines}

    assert by_id["E1"].execution_status == "fail_closed"
    assert "static" in by_id["E1"].non_executable_reason.lower()
    assert {
        baseline.id
        for baseline in registry.baselines
        if baseline.execution_status == "runnable"
    } == {baseline.id for baseline in registry.baselines} - {"E1"}


def test_registry_quarantines_every_historical_target_dependent_method() -> None:
    policy = yaml.safe_load(
        Path("configs/evaluation/evidence_status.yaml").read_text(encoding="utf-8")
    )
    quarantined = {"A3", "B1", "C1-human", "C2-human", "E1", "E2", "E3"}

    assert set(policy["quarantined_methods"]) == quarantined
    assert all("target" in note.lower() for note in policy["quarantined_methods"].values())


def test_primary_contrast_families_exclude_quarantined_methods() -> None:
    policy = yaml.safe_load(
        Path("configs/evaluation/evidence_status.yaml").read_text(encoding="utf-8")
    )
    quarantined = set(policy["quarantined_methods"])

    for path in (
        Path("configs/evaluation/primary_contrasts.yaml"),
        Path("configs/evaluation/compact_baseline_contrasts.yaml"),
    ):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        endpoints = {
            str(contrast[field])
            for contrast in config["contrasts"]
            for field in ("reference", "treatment")
        }
        assert not endpoints & quarantined, f"{path} contains quarantined primary contrasts"


def test_soft_distillation_config_cannot_omit_official_loss_controls() -> None:
    with pytest.raises(ValidationError, match="requires lmbda and beta"):
        BaselineConfig.model_validate(
            {
                "id": "broken",
                "name": "broken-gkd",
                "family": "on_policy",
                "backend": "trl_gkd",
                "official_source": "huggingface/trl",
                "trajectory_source": "student",
            }
        )


def test_shared_smoke_configs_cover_the_complete_trainable_matrix() -> None:
    registry = load_baseline_registry(Path("configs/baselines.yaml"))
    by_id = {baseline.id: baseline for baseline in registry.baselines}
    sft = load_trl_run_spec(Path("configs/training/sft_smoke.yaml"))
    gkd = load_trl_run_spec(Path("configs/training/gkd_smoke.yaml"))
    opsd = load_opsd_run_spec(Path("configs/training/opsd_smoke.yaml"))

    for baseline_id in ("B1", "B2a", "B2b", "B2c", "B3"):
        assert by_id[override_trl_baseline(sft, baseline_id).baseline_id].backend == "trl_sft"
    for baseline_id in (
        "C1-human",
        "C1-best1",
        "C1-diverse4",
        "C2-human",
        "C2-best1",
        "C2-diverse4",
        "D1",
        "D2",
        "D3",
    ):
        assert by_id[override_trl_baseline(gkd, baseline_id).baseline_id].backend == "trl_gkd"
    for baseline_id in ("E2", "E3", "E4"):
        assert by_id[override_opsd_baseline(opsd, baseline_id).baseline_id].backend == "opsd"
