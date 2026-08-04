"""Fail-closed evidence-status policy for inferential evaluation artifacts."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvidencePolicy:
    """Separate post-hoc validity findings from immutable training identities."""

    name: str
    quarantined_methods: Mapping[str, str]
    claim_boundary: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvidencePolicy":
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported evidence policy schema")
        name = str(raw.get("policy", "")).strip()
        claim_boundary = str(raw.get("claim_boundary", "")).strip()
        quarantined = raw.get("quarantined_methods")
        if not name or not claim_boundary or not isinstance(quarantined, Mapping):
            raise ValueError("evidence policy is missing required fields")
        normalized = {
            str(method).strip(): str(reason).strip()
            for method, reason in quarantined.items()
        }
        if any(not method or not reason for method, reason in normalized.items()):
            raise ValueError("evidence policy quarantine entries must be non-empty")
        return cls(
            name=name,
            quarantined_methods=normalized,
            claim_boundary=claim_boundary,
        )

    @classmethod
    def from_path(cls, path: Path) -> "EvidencePolicy":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("evidence policy must be a mapping")
        return cls.from_mapping(raw)

    def validate_primary_contrasts(
        self, contrasts: Iterable[Mapping[str, Any]]
    ) -> None:
        """Reject quarantined endpoints before any inferential computation."""

        for contrast in contrasts:
            contrast_id = str(contrast.get("id", "<unnamed>"))
            for endpoint in ("reference", "treatment"):
                method = str(contrast.get(endpoint, "")).strip()
                if method in self.quarantined_methods:
                    raise ValueError(
                        f"{method} is quarantined and cannot enter primary contrast "
                        f"{contrast_id}: {self.quarantined_methods[method]}"
                    )


def annotate_method_summaries(
    *, summaries: Mapping[str, Mapping[str, Any]], policy: EvidencePolicy
) -> dict[str, dict[str, Any]]:
    """Attach an explicit evidence status while retaining descriptive artifacts."""

    annotated: dict[str, dict[str, Any]] = {}
    for method, summary in summaries.items():
        values = dict(summary)
        if method in policy.quarantined_methods:
            values["evidence_status"] = "quarantined"
            values["evidence_note"] = policy.quarantined_methods[method]
        else:
            values["evidence_status"] = "primary_eligible"
        annotated[str(method)] = values
    return annotated
