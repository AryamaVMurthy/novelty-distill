"""Thin configuration plans for official baseline implementations."""

from dataclasses import dataclass
from typing import Any

from novelty_distill.config import BaselineConfig


@dataclass(frozen=True)
class TrainingBackendPlan:
    """Official checkout/API and the project-specific arguments passed to it."""

    repository: str
    entrypoint: str
    arguments: dict[str, Any]


_OFFICIAL_ENTRYPOINTS = {
    "trl_sft": ("trl", "trl.SFTTrainer"),
    "trl_gkd": ("trl", "trl.experimental.gkd.GKDTrainer"),
    "opsd": ("opsd", "opsd_train.py"),
    "gem": ("gem", "train.py"),
    "distillm": ("distillm", "finetune.py"),
}


def build_backend_plan(baseline: BaselineConfig) -> TrainingBackendPlan:
    """Translate one registry row without reimplementing its official trainer."""

    if baseline.backend == "evaluation":
        raise ValueError(f"{baseline.id} is an evaluation control, not a training baseline")
    repository, entrypoint = _OFFICIAL_ENTRYPOINTS[baseline.backend]
    arguments: dict[str, Any] = {
        "trajectory_source": baseline.trajectory_source,
        "target_view": baseline.target_view,
        "teacher_context": baseline.teacher_context,
        "divergence": baseline.divergence,
    }
    if baseline.lmbda is not None:
        arguments["lmbda"] = baseline.lmbda
    if baseline.beta is not None:
        arguments["beta"] = baseline.beta
    return TrainingBackendPlan(
        repository=repository,
        entrypoint=entrypoint,
        arguments=arguments,
    )
