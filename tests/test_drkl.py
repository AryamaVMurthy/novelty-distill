import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from novelty_distill.config import BaselineConfig, load_baseline_registry


def test_exploratory_drkl_registry_is_separate_and_complete() -> None:
    registry = load_baseline_registry(Path("configs/exploratory_baselines.yaml"))

    assert {item.id for item in registry.baselines} == {"F1-best1", "F1-diverse4"}
    assert all(item.divergence == "diversity_aware_reverse_kl" for item in registry.baselines)
    assert all(item.drkl_gamma == 0.5 for item in registry.baselines)


def test_drkl_gamma_is_rejected_for_unrelated_losses() -> None:
    with pytest.raises(ValidationError, match="only valid"):
        BaselineConfig.model_validate(
            {
                "id": "invalid",
                "name": "invalid",
                "family": "off_policy",
                "backend": "trl_gkd",
                "official_source": "test",
                "trajectory_source": "teacher",
                "target_view": "best1",
                "divergence": "reverse_kl",
                "lmbda": 0.0,
                "beta": 1.0,
                "drkl_gamma": 0.5,
            }
        )


def test_drkl_matches_direct_target_non_target_definition_and_backpropagates() -> None:
    torch = pytest.importorskip("torch")
    from novelty_distill.training.trl import diversity_aware_reverse_kl_loss

    student_probs = torch.tensor([[[0.55, 0.30, 0.15], [0.20, 0.50, 0.30]]])
    teacher_probs = torch.tensor([[[0.45, 0.35, 0.20], [0.25, 0.45, 0.30]]])
    student_logits = student_probs.log().requires_grad_(True)
    teacher_logits = teacher_probs.log()
    labels = torch.tensor([[0, 1]])
    gamma = 0.5

    actual = diversity_aware_reverse_kl_loss(
        student_logits, teacher_logits, labels, gamma=gamma, reduction="none"
    )

    expected = []
    for q, p, target in zip(student_probs[0], teacher_probs[0], labels[0], strict=True):
        target_index = int(target)
        q_target = float(q[target_index])
        p_target = float(p[target_index])
        binary = q_target * math.log(q_target / p_target) + (1 - q_target) * math.log(
            (1 - q_target) / (1 - p_target)
        )
        q_non = [float(value / (1 - q_target)) for i, value in enumerate(q) if i != target_index]
        p_non = [float(value / (1 - p_target)) for i, value in enumerate(p) if i != target_index]
        non_target = sum(qi * math.log(qi / pi) for qi, pi in zip(q_non, p_non, strict=True))
        expected.append(binary + gamma * non_target)

    assert actual.tolist() == pytest.approx(expected, abs=1e-6)
    actual.mean().backward()
    assert torch.isfinite(student_logits.grad).all()


def test_drkl_ignores_masked_positions() -> None:
    torch = pytest.importorskip("torch")
    from novelty_distill.training.trl import diversity_aware_reverse_kl_loss

    student_logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    labels = torch.tensor([[0, -100]])

    loss = diversity_aware_reverse_kl_loss(student_logits, teacher_logits, labels)
    loss.backward()

    assert student_logits.grad[0, 1].abs().sum().item() == 0
