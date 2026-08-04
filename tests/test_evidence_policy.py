import pytest

from novelty_distill.evaluation.evidence_policy import EvidencePolicy


def test_quarantined_method_cannot_enter_primary_contrast() -> None:
    policy = EvidencePolicy(
        name="test-v1",
        quarantined_methods={"A3": "contaminated target"},
        claim_boundary="descriptive only",
    )

    with pytest.raises(ValueError, match="A3.*quarantined"):
        policy.validate_primary_contrasts(
            ({"id": "A3-vs-A0", "reference": "A0", "treatment": "A3"},)
        )


def test_policy_mapping_is_strict() -> None:
    with pytest.raises(ValueError, match="schema"):
        EvidencePolicy.from_mapping(
            {
                "schema_version": 2,
                "policy": "test-v1",
                "quarantined_methods": {},
                "claim_boundary": "none",
            }
        )
