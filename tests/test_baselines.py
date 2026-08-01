from pathlib import Path

from novelty_distill.config import load_baseline_registry


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
