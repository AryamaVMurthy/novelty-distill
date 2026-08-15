import pytest
from pydantic import ValidationError

from novelty_distill.data.semantic_seeds import (
    GaussianSeedSpec,
    condition_prompt,
    gaussian_seed_values,
    render_seed_prefix,
)


def _spec() -> GaussianSeedSpec:
    return GaussianSeedSpec(dimensions=4, bins=5, scale=1.0, salt="v1")


def test_gaussian_seed_is_reproducible_and_sample_specific() -> None:
    first = gaussian_seed_values(
        prompt_id="paper-1", sample_index=0, generation_seed=17, spec=_spec()
    )
    repeat = gaussian_seed_values(
        prompt_id="paper-1", sample_index=0, generation_seed=17, spec=_spec()
    )
    second_sample = gaussian_seed_values(
        prompt_id="paper-1", sample_index=1, generation_seed=17, spec=_spec()
    )

    assert first == (2, 1, -1, -2)
    assert repeat == first
    assert second_sample != first


def test_gaussian_seed_quantization_is_bounded() -> None:
    values = gaussian_seed_values(
        prompt_id="paper-2",
        sample_index=9,
        generation_seed=43,
        spec=GaussianSeedSpec(dimensions=32, bins=9, scale=4.0, salt="pilot"),
    )

    assert len(values) == 32
    assert min(values) >= -4
    assert max(values) <= 4


def test_seed_prefix_contains_no_task_or_target_text() -> None:
    prefix = render_seed_prefix((-2, -1, 0, 1, 2))

    assert prefix == (
        "Exploration seed: N2-N1-Z0-P1-P2\n"
        "Use this arbitrary seed only to choose one coherent approach; do not mention it."
    )
    assert "hypothesis" not in prefix.lower()
    assert "teacher" not in prefix.lower()


def test_same_seed_renders_identically_for_training_and_generation() -> None:
    prompt = "Research question: Can this catalyst be stabilized?"
    values = gaussian_seed_values(
        prompt_id="paper-3", sample_index=2, generation_seed=29, spec=_spec()
    )

    assert condition_prompt(prompt, values=values) == (
        f"{render_seed_prefix(values)}\n\n{prompt}"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dimensions", 0),
        ("dimensions", 33),
        ("bins", 1),
        ("bins", 10),
        ("scale", 0),
        ("scale", 4.1),
        ("salt", ""),
    ],
)
def test_invalid_dimensions_scale_and_bins_fail_closed(field: str, value: object) -> None:
    payload = {"dimensions": 4, "bins": 5, "scale": 1.0, "salt": "v1"}
    payload[field] = value

    with pytest.raises(ValidationError):
        GaussianSeedSpec.model_validate(payload)


def test_seed_inputs_must_be_nonempty_and_indices_nonnegative() -> None:
    with pytest.raises(ValueError, match="prompt ID"):
        gaussian_seed_values(
            prompt_id=" ", sample_index=0, generation_seed=17, spec=_spec()
        )
    with pytest.raises(ValueError, match="sample index"):
        gaussian_seed_values(
            prompt_id="paper-1", sample_index=-1, generation_seed=17, spec=_spec()
        )
    with pytest.raises(ValueError, match="generation seed"):
        gaussian_seed_values(
            prompt_id="paper-1", sample_index=0, generation_seed=-1, spec=_spec()
        )
