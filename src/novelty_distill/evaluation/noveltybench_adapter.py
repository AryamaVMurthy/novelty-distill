"""Minimal adapter giving the pinned NoveltyBench task independent request seeds."""

from logging import getLogger
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import Generate, Solver, TaskState, fork, generate, solver
from inspect_evals.novelty_bench.novelty_bench import (
    DATASET_PATH,
    EVAL_VERSION,
    NOVELTY_BENCH_DATASET_REVISION,
    novelty_bench_scorer,
    record_to_sample,
)
from inspect_evals.novelty_bench.utils import select_device
from inspect_evals.utils.huggingface import hf_dataset

logger = getLogger(__name__)


@solver
def independent_multiple_generator(*, k: int = 10, base_seed: int = 17) -> Solver:
    """Generate K parallel responses with deterministic, non-overlapping seeds."""

    if k <= 0 or base_seed < 0:
        raise ValueError("K must be positive and base_seed must be non-negative")
    generation_seeds = tuple(range(base_seed, base_seed + k))

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        del generate_fn
        output_states = await fork(
            state,
            [generate(seed=seed) for seed in generation_seeds],
        )
        if len(output_states) != k or any(output.output is None for output in output_states):
            raise ValueError("NoveltyBench did not return every independent generation")
        result_state = output_states[0]
        result_state.metadata["all_completions"] = [
            output.output.completion for output in output_states if output.output is not None
        ]
        result_state.metadata["generation_seeds"] = list(generation_seeds)
        result_state.metadata["generation_seed_policy"] = "independent-per-generation-seed-v1"
        return result_state

    return solve  # type: ignore[return-value]


@task
def novelty_bench_independent(
    dataset_split: Literal["curated", "wildchat"] = "curated",
    num_generations: int = 10,
    base_seed: int = 17,
    equivalence_threshold: float = 0.102,
    quality_model: Literal["small", "large"] = "small",
    shuffle: bool = False,
    quality_model_device: str | None = None,
    partition_model_device: str | None = None,
) -> Task:
    """Run pinned NoveltyBench scoring with an auditable independent seed schedule."""

    dataset = hf_dataset(
        path=DATASET_PATH,
        split=dataset_split,
        sample_fields=record_to_sample,
        shuffle=shuffle,
        revision=NOVELTY_BENCH_DATASET_REVISION,
    )
    metadata = EVAL_VERSION.to_metadata()
    metadata.update(
        {
            "sampling_adapter": "independent-per-generation-seed-v1",
            "sampling_base_seed": base_seed,
            "sampling_seed_stride": 1,
        }
    )
    return Task(
        dataset=dataset,
        solver=independent_multiple_generator(k=num_generations, base_seed=base_seed),
        scorer=novelty_bench_scorer(
            equivalence_threshold=equivalence_threshold,
            quality_model=quality_model,
            quality_model_device=select_device(quality_model_device),
            partition_model_device=select_device(partition_model_device),
        ),
        version=EVAL_VERSION.comparability_version,
        metadata=metadata,
    )
