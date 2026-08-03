#!/usr/bin/env python3
"""Submit promoted model evaluation, scale controls, taste, and analysis as one graph."""

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


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or "," in value:
        raise ValueError(f"manifest {key!r} must be a non-empty comma-free string")
    return value


def _submit(command: list[str], *, environment: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **(environment or {})},
        check=True,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in (completed.stderr, completed.stdout) if part)
    matches = re.findall(r"Submitted batch job ([1-9][0-9]*)", output)
    if not matches:
        raise RuntimeError(f"could not parse submitted job ID from {command[0]!r}")
    return matches[-1]


def _submit_model(environment: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(ROOT / "scripts/submit_model_evaluation.sh")],
        cwd=ROOT,
        env={**os.environ, **environment},
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("model evaluation launcher did not return its graph JSON") from error
    if not result.get("evaluation_final") or not result.get("taste_final"):
        raise RuntimeError("model evaluation launcher omitted final evaluation or taste job")
    return result


def _submit_official(environment: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(ROOT / "scripts/submit_official_model_evaluation.sh")],
        cwd=ROOT,
        env={**os.environ, **environment},
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("official evaluation launcher did not return its graph JSON") from error
    if not result.get("combined"):
        raise RuntimeError("official evaluation launcher omitted its combined-result job")
    return result


def _control_plan(
    *,
    label: str,
    score_id: str,
    output_id: str,
    teacher_score_id: str,
    target_name: str,
    teacher_score_job_id: int,
    target_gate_job_id: int,
) -> dict[str, Any]:
    for value in (score_id, output_id, teacher_score_id, target_name):
        if SAFE_ID.fullmatch(value) is None:
            raise ValueError(f"unsafe control identifier {value!r}")
    export = {
        "TEACHER_SCORE_ID": teacher_score_id,
        "STUDENT_SCORE_ID": score_id,
        "SCORE_NAMESPACE": "evaluation-scores",
        "TARGET_NAME": target_name,
        "OUTPUT_NAME": output_id,
    }
    if label == "A3":
        export.update({"TEACHER_SAMPLES_PER_PROMPT": "16", "STUDENT_SAMPLES_PER_PROMPT": "1"})
    dependency = f"afterok:{teacher_score_job_id}:{target_gate_job_id}"
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export.items())
    return {
        "score_id": score_id,
        "evaluation_id": output_id,
        "environment": export,
        "command": [
            "scripts/turing_submit.sh",
            "slurm/evaluate_student.sbatch",
            f"--dependency={dependency}",
            f"--export={export_arg}",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-score-id")
    parser.add_argument("--teacher-score-job-id", type=int, required=True)
    parser.add_argument("--control-a0-score-id")
    parser.add_argument("--control-a1-score-id")
    parser.add_argument("--control-a3-score-id", default="A3-temporal-k1")
    parser.add_argument("--control-a0-taste-id")
    parser.add_argument("--control-a1-taste-id")
    parser.add_argument("--control-a3-taste-id", default="A3-temporal-k1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    profile = manifest.get("model_profile")
    train_size = manifest.get("train_size")
    if profile not in {"main", "replication"} or not isinstance(train_size, int):
        raise ValueError("manifest has invalid model profile or training size")
    run_infix = f"tomato{train_size}" if profile == "main" else f"qwen1p7b-tomato{train_size}"
    target_gate_job_id = manifest.get("target_gate_job_id")
    if not isinstance(target_gate_job_id, int) or target_gate_job_id <= 0:
        raise ValueError("manifest requires a positive target_gate_job_id")
    target_name = _required_text(manifest, "target_name")

    if profile == "main":
        teacher_score_id = args.teacher_score_id or "A1-temporal-k16-seed17000"
        a0_score_id = args.control_a0_score_id or "A0-temporal-k16-seed17000"
        a1_score_id = args.control_a1_score_id or teacher_score_id
        a0_taste_id = args.control_a0_taste_id or "A0-temporal-k16-seed17000"
        a1_taste_id = args.control_a1_taste_id or "A1-temporal-k16-seed17000"
    else:
        explicit = {
            "--teacher-score-id": args.teacher_score_id,
            "--control-a0-score-id": args.control_a0_score_id,
            "--control-a1-score-id": args.control_a1_score_id,
            "--control-a0-taste-id": args.control_a0_taste_id,
            "--control-a1-taste-id": args.control_a1_taste_id,
        }
        missing = [name for name, value in explicit.items() if not value]
        if missing:
            raise ValueError(
                "replication requires explicit control artifacts: " + ", ".join(missing)
            )
        teacher_score_id = args.teacher_score_id
        a0_score_id = args.control_a0_score_id
        a1_score_id = args.control_a1_score_id
        a0_taste_id = args.control_a0_taste_id
        a1_taste_id = args.control_a1_taste_id
    assert teacher_score_id and a0_score_id and a1_score_id and a0_taste_id and a1_taste_id

    control_specs = {
        "A0": (a0_score_id, f"A0-{run_infix}-temporal-k16"),
        "A1": (a1_score_id, f"A1-{run_infix}-temporal-k16"),
        "A3": (args.control_a3_score_id, f"A3-{run_infix}-temporal-k1"),
    }
    controls = {
        label: _control_plan(
            label=label,
            score_id=score_id,
            output_id=output_id,
            teacher_score_id=teacher_score_id,
            target_name=target_name,
            teacher_score_job_id=args.teacher_score_job_id,
            target_gate_job_id=target_gate_job_id,
        )
        for label, (score_id, output_id) in control_specs.items()
    }

    runs_payload = manifest.get("runs")
    if not isinstance(runs_payload, list) or not runs_payload:
        raise ValueError("manifest runs must be a non-empty list")
    planned_runs: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for run in runs_payload:
        baseline_id = _required_text(run, "baseline_id")
        seed = run.get("seed")
        if not isinstance(seed, int) or (baseline_id, seed) in identities:
            raise ValueError("manifest contains an invalid or duplicate baseline/seed run")
        identities.add((baseline_id, seed))
        dependency = _required_text(run, "training_dependency")
        if JOB_ID.fullmatch(dependency) is None:
            raise ValueError("training dependency is not a Slurm job or array-task ID")
        environment = {
            "EVAL_ID": _required_text(run, "evaluation_id"),
            "MODEL_PATH": _required_text(run, "model_path"),
            "MODEL_REVISION": _required_text(run, "model_revision"),
            "SERVED_MODEL_NAME": _required_text(run, "served_model_name"),
            "MODEL_DTYPE": _required_text(run, "model_dtype"),
            "GENERATION_CONFIG": _required_text(run, "generation_config"),
            "TRAINING_DEPENDENCY": dependency,
            "TEACHER_SCORE_ID": teacher_score_id,
            "TEACHER_SCORE_JOB_ID": str(args.teacher_score_job_id),
            "TARGET_JOB_ID": str(target_gate_job_id),
            "TARGET_NAME": target_name,
            "RUN_TASTE": "1",
        }
        lora_path = run.get("lora_path")
        if lora_path is not None:
            if not isinstance(lora_path, str) or not lora_path or "," in lora_path:
                raise ValueError("manifest lora_path must be null or a comma-free string")
            environment.update({"LORA_NAME": "student-adapter", "LORA_PATH": lora_path})
        planned_run = {
            "baseline_id": baseline_id,
            "seed": seed,
            "environment": environment,
            "command": ["scripts/submit_model_evaluation.sh"],
        }
        if args.run_official:
            official_environment = {
                "EVAL_ID": environment["EVAL_ID"],
                "MODEL_PATH": environment["MODEL_PATH"],
                "MODEL_REVISION": environment["MODEL_REVISION"],
                "MODEL_DTYPE": environment["MODEL_DTYPE"],
            }
            if lora_path is not None:
                official_environment.update(
                    {"LORA_NAME": "novelty-model", "LORA_PATH": lora_path}
                )
            planned_run["official"] = {
                "environment": official_environment,
                "command": ["scripts/submit_official_model_evaluation.sh"],
                "dependency_source": "temporal.evaluation_final",
            }
        planned_runs.append(planned_run)

    control_taste_ids = {
        "A0": a0_taste_id,
        "A1": a1_taste_id,
        "A3": args.control_a3_taste_id,
    }
    methods = manifest.get("baseline_ids")
    seeds = manifest.get("seeds")
    if not isinstance(methods, list) or not methods or not isinstance(seeds, list) or not seeds:
        raise ValueError("manifest baseline_ids and seeds must be non-empty lists")
    analysis_environment = {
        "MODEL_PROFILE": profile,
        "TRAIN_SIZE": str(train_size),
        "PROMOTED_METHODS": ",".join(methods),
        "TRAIN_SEEDS": ",".join(str(seed) for seed in seeds),
        "RUN_OFFICIAL": "1" if args.run_official else "0",
        **{
            f"CONTROL_{label}_EVAL_ID": controls[label]["evaluation_id"]
            for label in ("A0", "A1", "A3")
        },
        **{
            f"CONTROL_{label}_TASTE_ID": control_taste_ids[label]
            for label in ("A0", "A1", "A3")
        },
    }
    analysis = {
        "environment": analysis_environment,
        "command": ["scripts/turing_submit.sh", "slurm/analyze_promoted_matrix.sbatch"],
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "dry_run": args.dry_run,
        "repository_commit": repository_commit(ROOT),
        "training_manifest": str(args.manifest.resolve()),
        "model_profile": profile,
        "train_size": train_size,
        "controls": controls,
        "runs": planned_runs,
        "analysis": analysis,
    }

    if not args.dry_run:
        terminal_jobs: list[str] = []
        for control in controls.values():
            control["job_id"] = _submit(control["command"])
            terminal_jobs.append(control["job_id"])
        for run in planned_runs:
            run["submitted"] = _submit_model(run["environment"])
            terminal_jobs.extend(
                (run["submitted"]["evaluation_final"], run["submitted"]["taste_final"])
            )
            if args.run_official:
                official = run["official"]
                official["environment"]["DEPENDENCY_JOB_ID"] = run["submitted"][
                    "evaluation_final"
                ]
                official["submitted"] = _submit_official(official["environment"])
                terminal_jobs.append(official["submitted"]["combined"])
        dependency = "afterok:" + ":".join(terminal_jobs)
        export_arg = "ALL," + ",".join(
            f"{key}={value}" for key, value in analysis_environment.items()
        )
        analysis["command"].extend((f"--dependency={dependency}", f"--export={export_arg}"))
        analysis["job_id"] = _submit(analysis["command"])

    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
