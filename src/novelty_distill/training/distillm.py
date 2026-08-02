"""Thin data and command adapters for the pinned official DistiLLM repository."""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from novelty_distill.config import BaselineConfig
from novelty_distill.data.tomato import CanonicalExample
from novelty_distill.training.provenance import atomic_json, file_provenance


class DistiLLMRawRow(TypedDict):
    instruction: str
    input: str
    output: str


class DistiLLMEncodedRow(TypedDict):
    prompt_ids: list[int]
    completion_ids: list[int]
    truncated_completion_tokens: int


TeacherTargets = Mapping[str, Mapping[str, Sequence[str]]]
DISTILLM_SEPARATOR_ID = 65535


class DistiLLMRunSpec(BaseModel):
    """A bounded adaptive skew-FKL run through official `finetune.py`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_id: str
    student_model: str
    student_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    teacher_model: str
    teacher_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    input: Path
    teacher_targets: Path
    raw_dir: Path
    processed_dir: Path
    output_dir: Path
    max_examples: int = Field(gt=1)
    dev_examples: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_length: int = Field(gt=0)
    max_prompt_length: int = Field(gt=0)
    validation_interval: int = Field(gt=0)
    num_gpus: int = Field(default=1, gt=0)
    skew_alpha: float = Field(gt=0, lt=1)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def split_and_lengths_are_valid(self) -> "DistiLLMRunSpec":
        if self.dev_examples >= self.max_examples:
            raise ValueError("DistiLLM needs at least one train row after its validation prefix")
        train_examples = self.max_examples - self.dev_examples
        distributed_batch = self.batch_size * self.num_gpus
        if train_examples < distributed_batch:
            raise ValueError(
                "DistiLLM needs at least one complete distributed batch: "
                f"train_examples={train_examples}, distributed_batch={distributed_batch}"
            )
        if train_examples % distributed_batch:
            raise ValueError(
                "DistiLLM train rows must divide into complete distributed batches: "
                f"train_examples={train_examples}, distributed_batch={distributed_batch}"
            )
        if self.max_prompt_length >= self.max_length:
            raise ValueError("max_prompt_length must be smaller than max_length")
        return self


def build_distillm_raw_rows(
    baseline: BaselineConfig,
    examples: Sequence[CanonicalExample],
    *,
    teacher_targets: TeacherTargets,
) -> tuple[DistiLLMRawRow, ...]:
    """Map static best-one targets to the official Dolly-style raw JSON schema."""

    if baseline.backend != "distillm" or baseline.target_view != "best1":
        raise ValueError(f"baseline {baseline.id} is not the configured DistiLLM baseline")
    rows: list[DistiLLMRawRow] = []
    for example in examples:
        try:
            targets = tuple(teacher_targets[example.id]["best1"])
        except KeyError as error:
            raise ValueError(f"missing best1 teacher target for {example.id}") from error
        if len(targets) != 1 or not targets[0].strip():
            raise ValueError(f"best1 requires exactly one non-empty target for {example.id}")
        rows.append(
            {
                "instruction": example.student_prompt,
                "input": "",
                "output": targets[0].strip(),
            }
        )
    return tuple(rows)


def encode_distillm_chat_row(
    row: DistiLLMRawRow,
    *,
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
) -> DistiLLMEncodedRow:
    """Render one task-faithful Qwen row for the official indexed-data loader."""

    if row["input"]:
        raise ValueError("TOMATO DistiLLM rows must keep all context in the instruction")
    prompt_messages = [{"role": "user", "content": row["instruction"]}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": row["output"]},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rendered_full = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = list(
        tokenizer(
            rendered_prompt,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
    )
    full_ids = list(
        tokenizer(
            rendered_full,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Qwen full chat does not preserve the generation prompt prefix")
    if len(prompt_ids) > max_prompt_length:
        raise ValueError(
            f"DistiLLM prompt length {len(prompt_ids)} exceeds {max_prompt_length}"
        )
    completion_ids = full_ids[len(prompt_ids) :]
    completion_budget = max_length - len(prompt_ids)
    kept_completion = completion_ids[:completion_budget]
    if not kept_completion:
        raise ValueError("DistiLLM row has no assistant completion tokens")
    if DISTILLM_SEPARATOR_ID in prompt_ids or DISTILLM_SEPARATOR_ID in kept_completion:
        raise ValueError(
            "Qwen content contains DistiLLM's legacy separator token 65535"
        )
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": kept_completion,
        "truncated_completion_tokens": len(completion_ids) - len(kept_completion),
    }


def build_distillm_training_command(
    spec: DistiLLMRunSpec,
    *,
    python_executable: Path,
    official_checkout: Path,
    student_model_path: Path,
    teacher_model_path: Path,
    processed_dir: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    """Build the official adaptive skew-forward-KL launch command."""

    epoch_plan = distillm_epoch_plan(spec)

    return (
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={spec.num_gpus}",
        str(official_checkout / "finetune.py"),
        "--base-path",
        str(official_checkout),
        "--model-path",
        str(student_model_path),
        "--teacher-model-path",
        str(teacher_model_path),
        "--ckpt-name",
        "qwen3-student",
        "--teacher-ckpt-name",
        "qwen3-teacher",
        "--model-type",
        "qwen",
        "--teacher-model-type",
        "qwen",
        "--n-gpu",
        str(spec.num_gpus),
        "--data-dir",
        f"{processed_dir}{os.sep}",
        "--train-num",
        str(spec.max_examples - spec.dev_examples),
        "--dev-num",
        str(spec.dev_examples),
        "--num-workers",
        "0",
        "--lr",
        str(spec.learning_rate),
        "--lr-min",
        str(spec.learning_rate),
        "--batch-size",
        str(spec.batch_size),
        "--eval-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.01",
        "--clip-grad",
        "1.0",
        "--epochs",
        str(epoch_plan["epochs"]),
        "--total-iters",
        str(spec.max_steps),
        "--kd-ratio",
        "1.0",
        "--max-length",
        str(spec.max_length),
        "--max-prompt-length",
        str(spec.max_prompt_length),
        "--do-train",
        "--do-valid",
        "--save-interval",
        str(spec.max_steps),
        "--eval-interval",
        str(spec.validation_interval),
        "--log-interval",
        "1",
        "--mid-log-num",
        "1",
        "--save",
        str(output_dir),
        "--seed",
        str(spec.seed),
        "--deepspeed",
        "--deepspeed_config",
        str(
            official_checkout
            / "configs"
            / "deepspeed"
            / "ds_config_zero2_offload.json"
        ),
        "--type",
        "adaptive-sfkl",
        "--student-gen",
        "--do-sample",
        "--top-k",
        "0",
        "--top-p",
        "1.0",
        "--temperature",
        "1.0",
        "--gen-num-beams",
        "1",
        "--gen-top-p",
        "1.0",
        "--skew-alpha",
        str(spec.skew_alpha),
        "--init-threshold",
        "0.0",
        "--loss-eps",
        "0.1",
        "--capacity",
        "1000",
    )


def distillm_epoch_plan(spec: DistiLLMRunSpec) -> dict[str, int]:
    """Plan enough official sampler epochs to reach the requested save step."""

    train_examples = spec.max_examples - spec.dev_examples
    distributed_batch = spec.batch_size * spec.num_gpus
    steps_per_epoch = train_examples // distributed_batch
    return {
        "train_examples": train_examples,
        "steps_per_epoch": steps_per_epoch,
        "epochs": math.ceil(spec.max_steps / steps_per_epoch),
    }


def audit_distillm_log(
    path: Path, *, expected_steps: int, loss_epsilon: float = 0.1
) -> dict[str, object]:
    """Validate official progress and reconstruct its adaptive scheduler state."""

    if not path.is_file():
        raise ValueError(f"DistiLLM did not produce its official training log: {path}")
    global_steps: list[int] = []
    validation_losses: list[float] = []
    logged_thresholds: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := re.search(r"global iter:\s*(\d+)/\s*(\d+)", line):
            global_steps.append(int(match.group(1)))
        if line.startswith("dev |"):
            loss_match = re.search(r"avg_loss:\s*([^ |]+)", line)
            threshold_match = re.search(r"threshold:\s*([^ |]+)", line)
            if loss_match is None or threshold_match is None:
                raise ValueError(f"malformed DistiLLM validation log line: {line}")
            validation_losses.append(float(loss_match.group(1)))
            logged_thresholds.append(float(threshold_match.group(1)))
    unique_steps = sorted(set(global_steps))
    if unique_steps != list(range(1, expected_steps + 1)):
        raise ValueError(
            "DistiLLM did not log every requested optimizer step: "
            f"expected 1..{expected_steps}, observed {unique_steps[:1]}..{unique_steps[-1:]}"
        )
    if not validation_losses:
        raise ValueError("DistiLLM did not log its initial validation loss")
    previous_loss = validation_losses[0]
    terminal_threshold = logged_thresholds[0]
    for loss, logged_threshold in zip(
        validation_losses[1:], logged_thresholds[1:], strict=True
    ):
        if logged_threshold != terminal_threshold:
            raise ValueError(
                "DistiLLM logged a threshold inconsistent with its scheduler history"
            )
        if loss >= previous_loss + loss_epsilon:
            previous_loss = loss
            terminal_threshold = min(terminal_threshold + 0.1, 1.0)
    return {
        "logged_training_steps": len(unique_steps),
        "last_global_step": unique_steps[-1],
        "validation_checks": len(validation_losses),
        "validation_losses": validation_losses,
        "adaptive_thresholds": logged_thresholds,
        "terminal_adaptive_threshold": terminal_threshold,
    }


def load_distillm_run_spec(path: Path) -> DistiLLMRunSpec:
    with path.open(encoding="utf-8") as handle:
        return DistiLLMRunSpec.model_validate(yaml.safe_load(handle))


def normalize_distillm_qwen_sentinels(processed_dir: Path) -> int:
    """Map Qwen's uint32 -1 separator to the value expected by its loader."""

    replacements = 0
    source = (2**32 - 1).to_bytes(4, "little")
    target = (2**16 - 1).to_bytes(4, "little")
    for data_path in sorted(processed_dir.glob("*_*.bin")):
        payload = data_path.read_bytes()
        if len(payload) % 4:
            raise ValueError(f"DistiLLM uint32 data is misaligned: {data_path}")
        count = payload.count(source)
        if not count:
            continue
        updated = payload.replace(source, target)
        with data_path.open("r+b") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        replacements += count
    return replacements


def write_distillm_indexed_data(
    rows: Sequence[DistiLLMRawRow],
    *,
    tokenizer: Any,
    processed_dir: Path,
    official_checkout: Path,
    dev_examples: int,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, int | float | str]:
    """Write task-faithful rows with DistiLLM's official mmap-index builder."""

    import numpy as np
    import torch

    sys.path.insert(0, str(official_checkout))
    try:
        from data_utils.indexed_dataset import make_builder
    finally:
        sys.path.pop(0)

    encoded = tuple(
        encode_distillm_chat_row(
            row,
            tokenizer=tokenizer,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )
        for row in rows
    )
    qwen_dir = processed_dir / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "valid": (rows[:dev_examples], encoded[:dev_examples]),
        "train": (rows[dev_examples:], encoded[dev_examples:]),
    }
    for split, (split_rows, split_encoded) in splits.items():
        builder = make_builder(
            str(qwen_dir / f"{split}_0.bin"), impl="mmap", dtype=np.uint32
        )
        for item in split_encoded:
            builder.add_item(
                torch.IntTensor(
                    [
                        *item["prompt_ids"],
                        -1,
                        *item["completion_ids"],
                    ]
                )
            )
        builder.finalize(str(qwen_dir / f"{split}_0.idx"))
        _write_raw_jsonl(split_rows, qwen_dir / f"{split}.jsonl")

    truncated_tokens = [item["truncated_completion_tokens"] for item in encoded]
    return {
        "data_adapter": "qwen_official_chat_nonthinking",
        "rows": len(rows),
        "max_prompt_tokens": max(len(item["prompt_ids"]) for item in encoded),
        "max_untruncated_chat_tokens": max(
            len(item["prompt_ids"])
            + len(item["completion_ids"])
            + item["truncated_completion_tokens"]
            for item in encoded
        ),
        "truncated_rows": sum(value > 0 for value in truncated_tokens),
        "truncated_row_rate": sum(value > 0 for value in truncated_tokens) / len(rows),
        "truncated_completion_tokens": sum(truncated_tokens),
        "separator_id": DISTILLM_SEPARATOR_ID,
    }


def execute_distillm_training(
    spec: DistiLLMRunSpec,
    *,
    scratch_root: Path,
    registry_path: Path,
    manifest_path: Path,
    official_root: Path,
) -> dict[str, object]:
    """Run official Qwen preprocessing followed by official adaptive skew-FKL training."""

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from novelty_distill.config import load_baseline_registry
    from novelty_distill.official import (
        checkout_official_repository,
        load_official_repositories,
    )
    from novelty_distill.training.gem import load_teacher_targets
    from novelty_distill.training.trl import load_canonical_examples

    registry = load_baseline_registry(registry_path)
    matches = [baseline for baseline in registry.baselines if baseline.id == spec.baseline_id]
    if len(matches) != 1 or matches[0].backend != "distillm":
        raise ValueError(f"baseline {spec.baseline_id} is not a unique DistiLLM baseline")
    baseline = matches[0]
    repositories = load_official_repositories(manifest_path)
    checkout = checkout_official_repository(repositories["distillm"], official_root)

    input_path = _resolve_under(scratch_root, spec.input)
    target_path = _resolve_under(scratch_root, spec.teacher_targets)
    raw_dir = _resolve_under(scratch_root, spec.raw_dir)
    processed_dir = _resolve_under(scratch_root, spec.processed_dir)
    output_dir = _resolve_under(scratch_root, spec.output_dir)
    examples = load_canonical_examples(input_path, limit=spec.max_examples)
    rows = build_distillm_raw_rows(
        baseline,
        examples,
        teacher_targets=load_teacher_targets(target_path),
    )
    _write_raw_jsonl(rows, raw_dir / "raw.jsonl")

    student_path = Path(
        snapshot_download(repo_id=spec.student_model, revision=spec.student_revision)
    )
    teacher_path = Path(
        snapshot_download(repo_id=spec.teacher_model, revision=spec.teacher_revision)
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(checkout),
        "CODE_BASE": "HF",
        "WANDB_DISABLED": "true",
    }
    tokenizer = AutoTokenizer.from_pretrained(student_path, padding_side="right")
    context_audit = write_distillm_indexed_data(
        rows,
        tokenizer=tokenizer,
        processed_dir=processed_dir,
        official_checkout=checkout,
        dev_examples=spec.dev_examples,
        max_length=spec.max_length,
        max_prompt_length=spec.max_prompt_length,
    )
    sentinel_replacements = normalize_distillm_qwen_sentinels(processed_dir / "qwen")
    if sentinel_replacements != len(rows):
        raise ValueError(
            "expected one DistiLLM Qwen separator per row; "
            f"normalized {sentinel_replacements} for {len(rows)} rows"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_distillm_training_command(
            spec,
            python_executable=Path(sys.executable),
            official_checkout=checkout,
            student_model_path=student_path,
            teacher_model_path=teacher_path,
            processed_dir=processed_dir / "qwen",
            output_dir=output_dir,
        ),
        check=True,
        cwd=checkout,
        env=environment,
    )
    final_dir = _latest_distillm_checkpoint(output_dir)
    log_audit = audit_distillm_log(
        output_dir / "log.txt", expected_steps=spec.max_steps
    )

    metadata: dict[str, object] = {
        "baseline_id": baseline.id,
        "backend": baseline.backend,
        "run_spec": spec.model_dump(mode="json"),
        "official_commit": repositories["distillm"].commit,
        "student_model": spec.student_model,
        "student_revision": spec.student_revision,
        "teacher_model": spec.teacher_model,
        "teacher_revision": spec.teacher_revision,
        "divergence": baseline.divergence,
        "trajectory_source": "adaptive static/student replay",
        "target_view": baseline.target_view,
        "skew_alpha": spec.skew_alpha,
        "dataset_revision": examples[0].dataset_revision,
        "example_ids": [example.id for example in examples],
        "training_artifacts": {
            "input": file_provenance(input_path),
            "teacher_targets": file_provenance(target_path),
            "registry": file_provenance(registry_path),
            "official_manifest": file_provenance(manifest_path),
        },
        "normalized_qwen_separators": sentinel_replacements,
        "context_audit": context_audit,
        "training_rows": len(rows),
        "optimizer_example_exposures": spec.max_steps * spec.batch_size * spec.num_gpus,
        "epoch_plan": distillm_epoch_plan(spec),
        "log_audit": log_audit,
        "max_steps": spec.max_steps,
        "num_gpus": spec.num_gpus,
        "seed": spec.seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": os.environ.get("NOVELTY_GIT_COMMIT"),
        "output_dir": str(output_dir),
        "final_dir": str(final_dir),
    }
    atomic_json(output_dir / "run_metadata.json", metadata)
    return metadata


def _latest_distillm_checkpoint(output_dir: Path) -> Path:
    """Return the latest official checkpoint, rejecting metadata-only runs."""

    deployable = []
    for candidate in output_dir.iterdir() if output_dir.is_dir() else ():
        if not candidate.is_dir() or not candidate.name.isdigit():
            continue
        weights = (
            *candidate.glob("*.safetensors"),
            *candidate.glob("pytorch_model*.bin"),
            *candidate.glob("*.safetensors.index.json"),
            *candidate.glob("pytorch_model*.bin.index.json"),
        )
        if (candidate / "config.json").is_file() and weights:
            deployable.append(candidate)
    if not deployable:
        raise ValueError(f"DistiLLM produced no deployable checkpoint under {output_dir}")
    return max(deployable, key=lambda path: int(path.name))


def _write_raw_jsonl(rows: Sequence[DistiLLMRawRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                json.dump(row, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _resolve_under(root: Path, path: Path) -> Path:
    root = root.resolve()
    candidate = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path must remain under scratch root {root}: {path}")
    return candidate
