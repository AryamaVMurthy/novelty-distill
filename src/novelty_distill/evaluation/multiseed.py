"""Checkpoint-seed-aware contrasts for balanced training replications."""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from novelty_distill.evaluation.matrix import collect_prompt_metric_rows
from novelty_distill.evaluation.statistics import paired_bootstrap

_T_975 = {
    1: 12.706204736,
    2: 4.30265273,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
}


def _seed_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) < 2:
        raise ValueError("checkpoint-seed inference requires at least two training seeds")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    critical = _T_975.get(len(values) - 1, 1.959963985)
    half_width = critical * standard_error
    return {
        "checkpoint_seed_n": len(values),
        "mean_difference": mean,
        "seed_standard_deviation": standard_deviation,
        "seed_standard_error": standard_error,
        "t_95_ci_low": mean - half_width,
        "t_95_ci_high": mean + half_width,
        "minimum_seed_difference": min(values),
        "maximum_seed_difference": max(values),
        "sign_consistency": {
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "positive": sum(value > 0 for value in values),
        },
    }


def analyze_checkpoint_seed_contrasts(
    *,
    evaluations: Mapping[str, Mapping[int, Mapping[str, Any]]],
    controls: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    contrasts: Sequence[Mapping[str, str]],
    metrics: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Analyze paired prompts within each checkpoint, then summarize across checkpoints."""

    expected_seeds = tuple(seeds)
    if (
        len(expected_seeds) < 2
        or len(expected_seeds) != len(set(expected_seeds))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds
        )
    ):
        raise ValueError("at least two unique non-negative training seeds are required")
    if not metrics or len(metrics) != len(set(metrics)) or any(not value for value in metrics):
        raise ValueError("unique metric names are required")
    if not contrasts:
        raise ValueError("at least one contrast is required")
    expected_set = set(expected_seeds)
    for method, by_seed in evaluations.items():
        if not method or set(by_seed) != expected_set:
            raise ValueError(f"method {method!r} does not have exactly the expected seeds")

    normalized_contrasts: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    available = set(evaluations) | set(controls)
    for contrast in contrasts:
        contrast_id = str(contrast.get("id", ""))
        reference = str(contrast.get("reference", ""))
        treatment = str(contrast.get("treatment", ""))
        if (
            not contrast_id
            or contrast_id in seen_ids
            or reference == treatment
            or reference not in available
            or treatment not in available
        ):
            raise ValueError(f"invalid contrast {contrast!r}")
        if reference in controls and treatment in controls:
            raise ValueError("a checkpoint-seed contrast must include at least one trained method")
        seen_ids.add(contrast_id)
        normalized_contrasts.append((contrast_id, reference, treatment))

    results: dict[str, dict[str, Any]] = {metric: {} for metric in metrics}
    for contrast_index, (contrast_id, reference, treatment) in enumerate(normalized_contrasts):
        per_metric: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
        for seed_index, checkpoint_seed in enumerate(expected_seeds):
            reference_payload = (
                controls[reference]
                if reference in controls
                else evaluations[reference][checkpoint_seed]
            )
            treatment_payload = (
                controls[treatment]
                if treatment in controls
                else evaluations[treatment][checkpoint_seed]
            )
            rows = collect_prompt_metric_rows(
                {"reference": reference_payload, "treatment": treatment_payload},
                metrics=metrics,
            )
            by_method = {
                method: {str(row["prompt_id"]): row for row in rows if row["method"] == method}
                for method in ("reference", "treatment")
            }
            prompt_ids = tuple(sorted(by_method["reference"]))
            if set(prompt_ids) != set(by_method["treatment"]):
                raise ValueError(f"contrast {contrast_id!r} does not have paired prompts")
            for metric_index, metric in enumerate(metrics):
                try:
                    estimate = paired_bootstrap(
                        reference=tuple(
                            float(by_method["reference"][prompt_id][metric])
                            for prompt_id in prompt_ids
                        ),
                        treatment=tuple(
                            float(by_method["treatment"][prompt_id][metric])
                            for prompt_id in prompt_ids
                        ),
                        samples=bootstrap_samples,
                        seed=seed + contrast_index * 10_000 + metric_index * 100 + seed_index,
                    )
                except KeyError as error:
                    raise ValueError(f"metric {metric!r} is absent for {contrast_id!r}") from error
                payload = asdict(estimate)
                if not math.isfinite(payload["effect_size"]):
                    payload["effect_size"] = None
                per_metric[metric].append({"checkpoint_seed": checkpoint_seed, **payload})
        for metric in metrics:
            seed_effects = [float(row["mean_difference"]) for row in per_metric[metric]]
            results[metric][contrast_id] = {
                "reference": reference,
                "treatment": treatment,
                **_seed_summary(seed_effects),
                "per_seed": per_metric[metric],
            }

    return {
        "schema_version": 1,
        "replication_unit": "training_checkpoint_seed",
        "prompt_pairing": "paired_within_each_checkpoint_seed",
        "seeds": list(expected_seeds),
        "metrics": list(metrics),
        "results": results,
    }
