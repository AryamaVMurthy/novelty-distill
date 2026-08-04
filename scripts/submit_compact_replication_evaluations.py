#!/usr/bin/env python3
"""Submit gated matched-K=4 evaluations for compact multi-seed replications."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from novelty_distill.provenance import repository_commit
from novelty_distill.training.provenance import atomic_json

ROOT = Path(__file__).resolve().parents[1]
JOB_ID = re.compile(r"^[1-9][0-9]*(?:_[0-9]+)?$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EXPECTED_METHODS = ("B2a", "B2b", "C1-best1", "C2-best1", "D1", "D2")
K4_CONFIG = "configs/generation/eval_qwen3_4b_lora_k4.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", default="29,43")
    parser.add_argument("--seed17-b2a-evaluation-job-id", required=True)
    parser.add_argument("--target-node", default="node01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("--seeds must be comma-separated integers") from error
    if not seeds or len(seeds) != len(set(seeds)) or any(seed not in {29, 43} for seed in seeds):
        raise ValueError("compact replication submission seeds must be unique values from 29,43")
    return seeds


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or "," in value or ".." in value:
        raise ValueError(f"manifest {key!r} must be a safe non-empty comma-free string")
    return value


def _export(environment: dict[str, str]) -> str:
    for key, value in environment.items():
        if not key or not value or "," in key or "," in value:
            raise ValueError("Slurm exports must be non-empty and comma-free")
    return "ALL," + ",".join(f"{key}={value}" for key, value in environment.items())


def _job_plan(
    script: str,
    *,
    target_node: str,
    dependency: str,
    environment: dict[str, str],
    array: str | None = None,
    time: str | None = None,
) -> dict[str, Any]:
    command = ["scripts/turing_submit.sh", script, f"--nodelist={target_node}"]
    if array is not None:
        command.append(f"--array={array}")
    if time is not None:
        command.append(f"--time={time}")
    command.extend((f"--dependency=afterok:{dependency}", f"--export={_export(environment)}"))
    return {
        "dependency": dependency,
        "environment": environment,
        "command": command,
    }


def _submit(plan: dict[str, Any], *, dependency: str | None = None) -> str:
    command = list(plan["command"])
    if dependency is not None:
        command = [
            f"--dependency=afterok:{dependency}" if value.startswith("--dependency=") else value
            for value in command
        ]
        plan["dependency"] = dependency
        plan["command"] = command
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ,
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = "\n".join(part for part in (completed.stderr, completed.stdout) if part)
    matches = re.findall(r"Submitted batch job ([1-9][0-9]*)", rendered)
    if not matches:
        raise RuntimeError(f"could not parse Slurm job ID for {plan['command'][1]}")
    plan["job_id"] = matches[-1]
    return matches[-1]


def main() -> None:
    args = _parse_args()
    seeds = _parse_seeds(args.seeds)
    if not re.fullmatch(r"node[0-9]{2}", args.target_node):
        raise ValueError("--target-node must have the form nodeNN")
    if JOB_ID.fullmatch(args.seed17_b2a_evaluation_job_id) is None:
        raise ValueError("seed-17 B2a dependency must be a Slurm job or array-task ID")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model_profile") != "main"
        or manifest.get("train_size") != 1000
        or tuple(manifest.get("baseline_ids", ())) != EXPECTED_METHODS
        or tuple(manifest.get("seeds", ())) != (17, 29, 43)
    ):
        raise ValueError("manifest is not the frozen six-method main-4B 1k replication")
    target_name = _required_text(manifest, "target_name")
    runs_payload = manifest.get("runs")
    if not isinstance(runs_payload, list):
        raise ValueError("manifest runs must be a list")
    indexed = {(run.get("baseline_id"), run.get("seed")): run for run in runs_payload}
    required = {(method, seed) for seed in seeds for method in EXPECTED_METHODS}
    if not required.issubset(indexed) or len(indexed) != len(runs_payload):
        raise ValueError("manifest has missing or duplicate method/seed runs")

    planned_runs: list[dict[str, Any]] = []
    for seed in seeds:
        for method in EXPECTED_METHODS:
            source = indexed[(method, seed)]
            training_dependency = _required_text(source, "training_dependency")
            if JOB_ID.fullmatch(training_dependency) is None:
                raise ValueError("training dependency is not a Slurm job or array-task ID")
            lora_path = _required_text(source, "lora_path")
            model_path = _required_text(source, "model_path")
            model_revision = _required_text(source, "model_revision")
            served_model_name = _required_text(source, "served_model_name")
            model_dtype = _required_text(source, "model_dtype")
            eval_id = f"{method}-tomato1k-seed{seed}-temporal-k4"
            if SAFE_ID.fullmatch(eval_id) is None:
                raise ValueError("constructed an unsafe evaluation ID")
            output_name = f"{eval_id}-corrected-v2"
            generation_env = {
                "MODEL_PATH": model_path,
                "MODEL_REVISION": model_revision,
                "SERVED_MODEL_NAME": served_model_name,
                "MODEL_DTYPE": model_dtype,
                "GENERATION_CONFIG": K4_CONFIG,
                "INPUT_NAME": "tomato-open-test-1658.jsonl",
                "OUTPUT_NAME": "evaluation-student",
                "OUTPUT_RUN_ID": eval_id,
                "INPUT_LIMIT": "1658",
                "GENERATION_CONCURRENCY": "8",
                "GENERATION_NUM_SHARDS": "4",
                "LORA_NAME": "student-adapter",
                "LORA_PATH": lora_path,
            }
            gate_env = {
                "GENERATION_NAME": "evaluation-student",
                "GENERATION_ID": eval_id,
                "GENERATION_CONFIG": K4_CONFIG,
                "INPUT_NAME": "tomato-open-test-1658.jsonl",
                "EXPECTED_PROMPTS": "1658",
                "SERVED_ARTIFACT_PATH": lora_path,
            }
            score_env = {
                "GENERATION_JOB_ID": eval_id,
                "GENERATION_NAME": "evaluation-student",
                "GENERATION_CONFIG": K4_CONFIG,
                "PROMPTS_NAME": "tomato-open-test-1658.jsonl",
                "SCORE_NAMESPACE": "evaluation-scores",
                "SCORE_CONCURRENCY": "8",
                "SCORE_NUM_SHARDS": "4",
            }
            score_gate_env = {
                "GENERATION_NAME": "evaluation-student",
                "GENERATION_ID": eval_id,
                "GENERATION_CONFIG": K4_CONFIG,
                "SCORE_NAMESPACE": "evaluation-scores",
                "EXPECTED_PROMPTS": "1658",
            }
            evaluation_env = {
                "TEACHER_SCORE_ID": "A1-temporal-k16-seed17000",
                "STUDENT_SCORE_ID": eval_id,
                "SCORE_NAMESPACE": "evaluation-scores",
                "TARGET_NAME": target_name,
                "OUTPUT_NAME": output_name,
                "TEACHER_SAMPLES_PER_PROMPT": "4",
                "STUDENT_SAMPLES_PER_PROMPT": "4",
                "TEACHER_SOURCE_SAMPLES_PER_PROMPT": "16",
                "STUDENT_SOURCE_SAMPLES_PER_PROMPT": "4",
            }
            planned_runs.append(
                {
                    "baseline_id": method,
                    "seed": seed,
                    "training_dependency": training_dependency,
                    "evaluation_id": eval_id,
                    "output_name": output_name,
                    "generation": _job_plan(
                        "slurm/sglang_smoke.sbatch",
                        target_node=args.target_node,
                        dependency=training_dependency,
                        environment=generation_env,
                        array="0-3%4",
                        time="03:00:00",
                    ),
                    "generation_gate": _job_plan(
                        "slurm/validate_generation_run.sbatch",
                        target_node=args.target_node,
                        dependency="<generation_job_id>",
                        environment=gate_env,
                    ),
                    "score": _job_plan(
                        "slurm/score_teacher.sbatch",
                        target_node=args.target_node,
                        dependency="<generation_gate_job_id>",
                        environment=score_env,
                        array="0-3%4",
                        time="03:00:00",
                    ),
                    "score_gate": _job_plan(
                        "slurm/validate_score_run.sbatch",
                        target_node=args.target_node,
                        dependency="<score_job_id>",
                        environment=score_gate_env,
                    ),
                    "evaluation": _job_plan(
                        "slurm/evaluate_student.sbatch",
                        target_node=args.target_node,
                        dependency="<score_gate_job_id>",
                        environment=evaluation_env,
                        time="03:00:00",
                    ),
                }
            )

    analysis_environment = {
        "TRAIN_SEEDS": "17,29,43",
        "COMPACT_METHODS": ",".join(EXPECTED_METHODS),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "dry_run": args.dry_run,
        "repository_commit": repository_commit(ROOT),
        "training_manifest": str(args.manifest.resolve()),
        "target_node": args.target_node,
        "seeds": list(seeds),
        "methods": list(EXPECTED_METHODS),
        "runs": planned_runs,
        "analysis": {
            "external_dependencies": [args.seed17_b2a_evaluation_job_id],
            "environment": analysis_environment,
            "command": [
                "scripts/turing_submit.sh",
                "slurm/analyze_compact_multiseed.sbatch",
                f"--nodelist={args.target_node}",
                "--dependency=afterok:<all_evaluation_jobs>",
                "--export=ALL",
            ],
        },
    }

    if not args.dry_run:
        evaluation_jobs = [args.seed17_b2a_evaluation_job_id]
        for run in planned_runs:
            generation_job = _submit(run["generation"])
            generation_gate_job = _submit(run["generation_gate"], dependency=generation_job)
            score_job = _submit(run["score"], dependency=generation_gate_job)
            score_gate_job = _submit(run["score_gate"], dependency=score_job)
            evaluation_jobs.append(
                _submit(run["evaluation"], dependency=score_gate_job)
            )
            atomic_json(args.output, result)
        analysis_dependency = ":".join(evaluation_jobs)
        analysis = result["analysis"]
        analysis["command"] = [
            f"--dependency=afterok:{analysis_dependency}"
            if value.startswith("--dependency=")
            else value
            for value in analysis["command"]
        ]
        analysis["job_id"] = _submit(analysis)

    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
