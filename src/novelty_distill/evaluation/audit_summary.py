"""Compact, content-bound summaries of corrected evaluation reruns."""

from collections.abc import Mapping
from typing import Any

_REQUIRED_QUALITY_GATE = {
    "relevance_minimum": 4,
    "feasibility_minimum": 4,
    "soundness_minimum": 4,
    "clarity_minimum": 4,
}


def _shared_controls(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    inputs = payload.get("inputs")
    embedding = payload.get("embedding")
    thresholds = payload.get("threshold_sensitivity")
    if not isinstance(inputs, Mapping) or not isinstance(embedding, Mapping):
        raise ValueError("corrected evaluation is missing provenance")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("corrected evaluation is missing threshold sensitivity")
    return (
        payload.get("schema_version"),
        payload.get("git_commit"),
        payload.get("num_prompts"),
        payload.get("teacher_samples_per_prompt"),
        payload.get("student_samples_per_prompt"),
        payload.get("primary_cosine_threshold"),
        inputs.get("teacher_score_sha256"),
        embedding.get("model"),
        embedding.get("revision"),
        tuple(thresholds),
    )


def summarize_corrected_evaluations(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    input_sha256: Mapping[str, str],
    jobs: Mapping[str, str],
) -> dict[str, Any]:
    """Validate compatible reruns and remove bulky per-prompt records."""

    if not payloads or set(payloads) != set(input_sha256) or set(payloads) != set(jobs):
        raise ValueError("payload, hash, and job method sets must be identical and non-empty")
    controls = {_shared_controls(payload) for payload in payloads.values()}
    if len(controls) != 1:
        raise ValueError("methods do not have shared evaluation controls")

    methods: dict[str, Any] = {}
    for method, payload in sorted(payloads.items()):
        if not method.strip():
            raise ValueError("method names must be non-empty")
        yield_spec = payload.get("quality_qualified_semantic_yield")
        if not isinstance(yield_spec, Mapping) or (
            yield_spec.get("status") != "corrected_secondary_descriptive"
            or yield_spec.get("quality_gate") != _REQUIRED_QUALITY_GATE
        ):
            raise ValueError("corrected yield must use the feasibility-inclusive quality gate")
        overall = payload.get("overall")
        threshold_sensitivity = payload.get("threshold_sensitivity")
        if not isinstance(overall, Mapping) or not isinstance(
            threshold_sensitivity, Mapping
        ):
            raise ValueError("corrected evaluation has no aggregate metrics")
        if "quality_qualified_semantic_yield" not in overall or any(
            not isinstance(metrics, Mapping)
            or "quality_qualified_semantic_yield" not in metrics
            for metrics in threshold_sensitivity.values()
        ):
            raise ValueError("corrected yield is absent from an aggregate or threshold")
        methods[method] = {
            "job_id": str(jobs[method]),
            "input_sha256": str(input_sha256[method]),
            "score_inputs": dict(payload["inputs"]),
            "overall": dict(overall),
            "threshold_sensitivity": {
                str(threshold): dict(metrics)
                for threshold, metrics in threshold_sensitivity.items()
            },
        }

    first = next(iter(payloads.values()))
    return {
        "schema_version": 1,
        "producer_commit": first["git_commit"],
        "num_prompts": first["num_prompts"],
        "teacher_samples_per_prompt": first["teacher_samples_per_prompt"],
        "student_samples_per_prompt": first["student_samples_per_prompt"],
        "primary_cosine_threshold": first["primary_cosine_threshold"],
        "embedding": dict(first["embedding"]),
        "quality_gate": dict(_REQUIRED_QUALITY_GATE),
        "metric_status": {
            "student_feasibility_mean": "operational_judge_primary_eligible",
            "student_soundness_mean": "operational_judge_primary_eligible",
            "quality_qualified_semantic_yield": "corrected_secondary_descriptive",
            "viable_semantic_yield": "legacy_reproducibility_only",
        },
        "single_threshold_inference_allowed": False,
        "claim_boundary": (
            "All embedding-threshold metrics are descriptive until the equivalence boundary "
            "is selected on blinded human labels. Judge metrics are operational and require "
            "different-family and human calibration for paper claims."
        ),
        "methods": methods,
    }
