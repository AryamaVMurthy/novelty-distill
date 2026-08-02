"""Validated experiment configuration."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class BaselineConfig(BaseModel):
    """One planned baseline and its official-implementation boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    family: Literal["control", "sft", "off_policy", "on_policy", "self_distillation"]
    backend: Literal["evaluation", "trl_sft", "trl_gkd", "gem", "distillm", "opsd"]
    official_source: str
    execution_status: Literal["runnable", "fail_closed"] = "runnable"
    non_executable_reason: str = ""
    trajectory_source: Literal["none", "human", "teacher", "student", "static"] = "none"
    target_view: Literal["none", "human", "random1", "best1", "mode1", "diverse4"] = "none"
    divergence: Literal["none", "forward_kl", "reverse_kl", "generalized_jsd", "skew_kl"] = (
        "none"
    )
    teacher_context: Literal["none", "ordinary", "privileged"] = "none"
    lmbda: float | None = None
    beta: float | None = None

    @model_validator(mode="after")
    def official_soft_loss_controls_are_complete(self) -> "BaselineConfig":
        if self.backend in {"trl_gkd", "opsd"} and (
            self.lmbda is None or self.beta is None
        ):
            raise ValueError(f"{self.backend} requires lmbda and beta")
        if self.trajectory_source == "student" and self.lmbda != 1.0:
            raise ValueError("student trajectories require fully on-policy lmbda=1")
        if self.execution_status == "fail_closed" and not self.non_executable_reason.strip():
            raise ValueError("fail-closed baselines require a non-executable reason")
        if self.execution_status == "runnable" and self.non_executable_reason:
            raise ValueError("runnable baselines cannot have a non-executable reason")
        return self


class BaselineRegistry(BaseModel):
    """Complete baseline registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    baselines: tuple[BaselineConfig, ...]

    @model_validator(mode="after")
    def baseline_ids_are_unique(self) -> "BaselineRegistry":
        ids = [baseline.id for baseline in self.baselines]
        if len(ids) != len(set(ids)):
            raise ValueError("baseline IDs must be unique")
        return self


def load_baseline_registry(path: Path) -> BaselineRegistry:
    """Load and validate the baseline registry at ``path``."""

    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return BaselineRegistry.model_validate(payload)
