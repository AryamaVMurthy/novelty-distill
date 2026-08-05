import pytest

from novelty_distill.evaluation.semantic_calibration import (
    HumanEquivalenceLabel,
    analyze_semantic_calibration,
    build_semantic_calibration_sample,
)


def _pairs() -> list[dict]:
    pairs = []
    bins = ((0.74, 0.84), (0.84, 0.90), (0.90, 0.94), (0.94, 1.001))
    for bin_index, (lower, upper) in enumerate(bins):
        for index in range(8):
            similarity = lower + (upper - lower) * (index + 1) / 10
            pairs.append(
                {
                    "prompt_id": f"p-{bin_index}-{index}",
                    "left_index": 0,
                    "right_index": 1,
                    "left_text": f"left answer {bin_index} {index}",
                    "right_text": f"right answer {bin_index} {index}",
                    "similarity": similarity,
                }
            )
            pairs.append(
                {
                    "prompt_id": f"p-{bin_index}-{index}",
                    "left_index": 0,
                    "right_index": 2,
                    "left_text": f"left answer {bin_index} {index}",
                    "right_text": f"alternate answer {bin_index} {index}",
                    "similarity": similarity + min((upper - similarity) / 2, 0.001),
                }
            )
    return pairs


def test_semantic_calibration_packet_is_balanced_blinded_and_repeatable() -> None:
    bins = ((0.74, 0.84), (0.84, 0.90), (0.90, 0.94), (0.94, 1.001))
    prompts = {pair["prompt_id"]: f"research prompt {pair['prompt_id']}" for pair in _pairs()}
    first = build_semantic_calibration_sample(
        pairs=_pairs(),
        prompts=prompts,
        similarity_bins=bins,
        pairs_per_bin=4,
        repeat_fraction=0.25,
        seed=17,
    )
    second = build_semantic_calibration_sample(
        pairs=_pairs(),
        prompts=prompts,
        similarity_bins=bins,
        pairs_per_bin=4,
        repeat_fraction=0.25,
        seed=17,
    )
    assert first == second
    assert len(first.private_entries) == 20
    assert len(first.public_entries) == 20
    originals = [entry for entry in first.private_entries if entry.repeat_of is None]
    assert len(originals) == 16
    assert {
        index: sum(entry.similarity_bin == index for entry in originals) for index in range(4)
    } == {
        0: 4,
        1: 4,
        2: 4,
        3: 4,
    }
    assert len({entry.prompt_id for entry in originals}) == 16
    public_text = str([entry.model_dump() for entry in first.public_entries])
    assert "similarity" not in public_text
    assert "prompt_id" not in public_text
    assert "repeat_of" not in public_text
    public_by_id = {entry.blind_id: entry for entry in first.public_entries}
    private_by_id = {entry.blind_id: entry for entry in first.private_entries}
    for repeated in (entry for entry in first.private_entries if entry.repeat_of):
        original_public = public_by_id[repeated.repeat_of]
        repeated_public = public_by_id[repeated.blind_id]
        assert repeated_public.answer_a == original_public.answer_b
        assert repeated_public.answer_b == original_public.answer_a
        assert private_by_id[repeated.repeat_of].similarity == repeated.similarity


def test_semantic_calibration_analysis_requires_adjudication_and_selects_threshold() -> None:
    sample = build_semantic_calibration_sample(
        pairs=_pairs(),
        prompts={pair["prompt_id"]: f"prompt {pair['prompt_id']}" for pair in _pairs()},
        similarity_bins=((0.74, 0.84), (0.84, 0.90), (0.90, 0.94), (0.94, 1.001)),
        pairs_per_bin=4,
        repeat_fraction=0.25,
        seed=23,
    )
    labels = {}
    for entry in sample.private_entries:
        source = (
            next(item for item in sample.private_entries if item.blind_id == entry.repeat_of)
            if entry.repeat_of
            else entry
        )
        label = "equivalent" if source.similarity >= 0.92 else "not_equivalent"
        labels[entry.blind_id] = HumanEquivalenceLabel(
            blind_id=entry.blind_id,
            label=label,
            confidence=3,
        )
    summary = analyze_semantic_calibration(
        entries=sample.private_entries,
        rater_one=labels,
        rater_two=labels,
        adjudicated={},
        threshold_grid=(0.88, 0.90, 0.92, 0.94, 0.96),
    )
    assert summary["selected_threshold"] == 0.92
    assert summary["inter_rater"]["cohen_kappa"] == 1
    assert summary["repeat_reliability"]["rater_one_exact_rate"] == 1
    assert summary["threshold_metrics"]["0.920"]["balanced_accuracy"] == 1

    rater_two = dict(labels)
    original = next(entry for entry in sample.private_entries if entry.repeat_of is None)
    rater_two[original.blind_id] = rater_two[original.blind_id].model_copy(
        update={"label": "uncertain"}
    )
    with pytest.raises(ValueError, match="adjudication"):
        analyze_semantic_calibration(
            entries=sample.private_entries,
            rater_one=labels,
            rater_two=rater_two,
            adjudicated={},
            threshold_grid=(0.90, 0.92, 0.94),
        )


def test_semantic_calibration_does_not_select_boundary_when_agreement_gate_fails() -> None:
    sample = build_semantic_calibration_sample(
        pairs=_pairs(),
        prompts={pair["prompt_id"]: f"prompt {pair['prompt_id']}" for pair in _pairs()},
        similarity_bins=((0.74, 0.84), (0.84, 0.90), (0.90, 0.94), (0.94, 1.001)),
        pairs_per_bin=4,
        repeat_fraction=0.25,
        seed=29,
    )
    rater_one = {}
    rater_two = {}
    adjudicated = {}
    for entry in sample.private_entries:
        source = (
            next(item for item in sample.private_entries if item.blind_id == entry.repeat_of)
            if entry.repeat_of
            else entry
        )
        first_label = "equivalent" if source.similarity >= 0.92 else "not_equivalent"
        second_label = "not_equivalent" if first_label == "equivalent" else "equivalent"
        rater_one[entry.blind_id] = HumanEquivalenceLabel(
            blind_id=entry.blind_id,
            label=first_label,
            confidence=3,
        )
        rater_two[entry.blind_id] = HumanEquivalenceLabel(
            blind_id=entry.blind_id,
            label=second_label,
            confidence=3,
        )
        if entry.repeat_of is None:
            adjudicated[entry.blind_id] = first_label

    summary = analyze_semantic_calibration(
        entries=sample.private_entries,
        rater_one=rater_one,
        rater_two=rater_two,
        adjudicated=adjudicated,
        threshold_grid=(0.88, 0.90, 0.92, 0.94, 0.96),
    )

    assert summary["inter_rater"]["passed"] is False
    assert summary["status"] == "failed_inter_rater_gate"
    assert summary["selected_threshold"] is None
    assert summary["candidate_threshold"] == 0.92
