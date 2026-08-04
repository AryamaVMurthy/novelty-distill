"""Validity policy for embedding-threshold-dependent semantic metrics."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SemanticValidityPolicy:
    name: str
    single_threshold_inference_allowed: bool
    full_threshold_curve_required: bool
    claim_boundary: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SemanticValidityPolicy":
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported semantic validity policy schema")
        policy = cls(
            name=str(raw.get("policy", "")).strip(),
            single_threshold_inference_allowed=bool(
                raw.get("single_threshold_inference_allowed", False)
            ),
            full_threshold_curve_required=bool(
                raw.get("full_threshold_curve_required", False)
            ),
            claim_boundary=str(raw.get("claim_boundary", "")).strip(),
        )
        if not policy.name or not policy.claim_boundary:
            raise ValueError("semantic validity policy is incomplete")
        return policy

    @classmethod
    def from_path(cls, path: Path) -> "SemanticValidityPolicy":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("semantic validity policy must be a mapping")
        return cls.from_mapping(raw)

    def validate_analysis_config(self, config: Mapping[str, Any]) -> None:
        """Prevent uncalibrated threshold metrics from entering primary inference."""

        metrics = {str(value) for value in config.get("metrics", ())}
        threshold_metrics = {
            str(value) for value in config.get("threshold_metric_directions", {})
        }
        if not self.single_threshold_inference_allowed and (
            overlap := sorted(metrics & threshold_metrics)
        ):
            raise ValueError(
                f"{', '.join(overlap)} are uncalibrated single-threshold metrics and "
                "cannot enter primary inference"
            )
        if self.full_threshold_curve_required and not threshold_metrics:
            raise ValueError("semantic validity policy requires full threshold-curve diagnostics")
