"""Tokenized-data adapter for the pinned official GEM trainer."""

import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, Field

from novelty_distill.config import BaselineConfig
from novelty_distill.data.tomato import CanonicalExample


class GEMTokenizedRow(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


TeacherTargets = Mapping[str, Mapping[str, Sequence[str]]]


class GEMRunSpec(BaseModel):
    """A bounded run delegated to the pinned official GEM entrypoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_id: str
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    input: Path
    teacher_targets: Path
    tokenized_output: Path
    output_dir: Path
    max_examples: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_length: int = Field(gt=0)
    gem_beta: float = Field(gt=0, lt=1)
    seed: int = Field(ge=0)


def install_gem_trainer_compat(trainer_class: type[Any]) -> bool:
    """Accept Transformers' new duplicate LR argument in official GEM's override."""

    original = trainer_class._maybe_log_save_evaluate
    if "learning_rate" in inspect.signature(original).parameters:
        return False

    def compatible(
        self: Any,
        tr_loss: Any,
        grad_norm: Any,
        model: Any,
        trial: Any,
        epoch: Any,
        ignore_keys_for_eval: Any,
        start_time: Any,
        learning_rate: Any = None,
    ) -> Any:
        del learning_rate
        return original(
            self,
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
        )

    trainer_class._maybe_log_save_evaluate = compatible
    return True


def build_official_gem_command(
    spec: GEMRunSpec,
    *,
    python_executable: Path,
    official_checkout: Path,
    model_path: Path,
    tokenized_path: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    """Build a one-process distributed command required by official GEM `train.py`."""

    return (
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--module",
        "novelty_distill.training.gem_compat",
        "--official-train",
        str(official_checkout / "train.py"),
        "--model_name_or_path",
        str(model_path),
        "--train_tokenized_file",
        str(tokenized_path),
        "--output_dir",
        str(output_dir),
        "--loss",
        "gem",
        "--gem_beta",
        str(spec.gem_beta),
        "--gem_h",
        "linear",
        "--use_flash_attn",
        "False",
        "--bf16",
        "True",
        "--gradient_checkpointing",
        "True",
        "--max_steps",
        str(spec.max_steps),
        "--per_device_train_batch_size",
        str(spec.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(spec.gradient_accumulation_steps),
        "--learning_rate",
        str(spec.learning_rate),
        "--max_seq_length",
        str(spec.max_length),
        "--logging_steps",
        "1",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(spec.max_steps),
        "--save_total_limit",
        "2",
        "--report_to",
        "none",
        "--seed",
        str(spec.seed),
        "--data_seed",
        str(spec.seed),
    )


def load_gem_run_spec(path: Path) -> GEMRunSpec:
    with path.open(encoding="utf-8") as handle:
        return GEMRunSpec.model_validate(yaml.safe_load(handle))


def execute_gem_training(
    spec: GEMRunSpec,
    *,
    scratch_root: Path,
    registry_path: Path,
    manifest_path: Path,
    official_root: Path,
) -> dict[str, object]:
    """Prepare B4 data and delegate optimization to the untouched official GEM trainer."""

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint

    from novelty_distill.config import load_baseline_registry
    from novelty_distill.official import (
        checkout_official_repository,
        load_official_repositories,
    )
    from novelty_distill.training.trl import load_canonical_examples

    registry = load_baseline_registry(registry_path)
    matches = [baseline for baseline in registry.baselines if baseline.id == spec.baseline_id]
    if len(matches) != 1 or matches[0].backend != "gem":
        raise ValueError(f"baseline {spec.baseline_id} is not a unique GEM baseline")
    baseline = matches[0]

    repositories = load_official_repositories(manifest_path)
    official_checkout = checkout_official_repository(repositories["gem"], official_root)
    input_path = _resolve_under(scratch_root, spec.input)
    targets_path = _resolve_under(scratch_root, spec.teacher_targets)
    tokenized_path = _resolve_under(scratch_root, spec.tokenized_output)
    output_dir = _resolve_under(scratch_root, spec.output_dir)
    examples = load_canonical_examples(input_path, limit=spec.max_examples)

    model_path = Path(snapshot_download(repo_id=spec.model, revision=spec.revision))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rows = build_gem_rows(
        baseline,
        examples,
        teacher_targets=load_teacher_targets(targets_path),
        tokenizer=tokenizer,
        max_length=spec.max_length,
    )
    write_gem_jsonl(rows, tokenized_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_official_gem_command(
        spec,
        python_executable=Path(sys.executable),
        official_checkout=official_checkout,
        model_path=model_path,
        tokenized_path=tokenized_path,
        output_dir=output_dir,
    )
    if checkpoint := get_last_checkpoint(str(output_dir)):
        command += ("--resume_from_checkpoint", checkpoint)
    subprocess.run(command, check=True)

    metrics_path = output_dir / "train_results.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metadata: dict[str, object] = {
        "baseline_id": baseline.id,
        "backend": baseline.backend,
        "official_commit": repositories["gem"].commit,
        "model": spec.model,
        "revision": spec.revision,
        "trajectory_source": baseline.trajectory_source,
        "target_view": baseline.target_view,
        "gem_beta": spec.gem_beta,
        "dataset_revision": examples[0].dataset_revision,
        "example_ids": [example.id for example in examples],
        "training_rows": len(rows),
        "optimizer_example_exposures": (
            spec.max_steps
            * spec.per_device_train_batch_size
            * spec.gradient_accumulation_steps
        ),
        "max_steps": spec.max_steps,
        "seed": spec.seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": os.environ.get("NOVELTY_GIT_COMMIT"),
        "metrics": metrics,
        "output_dir": str(output_dir),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def load_teacher_targets(path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """Load the versioned target-view artifact shared by teacher-trained baselines."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported teacher-target schema in {path}")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, Mapping):
        raise ValueError(f"teacher-target artifact has no target mapping: {path}")

    targets: dict[str, dict[str, tuple[str, ...]]] = {}
    for prompt_id, raw_views in raw_targets.items():
        if not isinstance(prompt_id, str) or not prompt_id or not isinstance(raw_views, Mapping):
            raise ValueError(f"invalid teacher-target prompt entry in {path}")
        views: dict[str, tuple[str, ...]] = {}
        for view, raw_values in raw_views.items():
            if (
                not isinstance(view, str)
                or not isinstance(raw_values, list)
                or not raw_values
                or any(not isinstance(value, str) or not value.strip() for value in raw_values)
            ):
                raise ValueError(f"invalid teacher target view {view!r} for {prompt_id}")
            views[view] = tuple(value.strip() for value in raw_values)
        targets[prompt_id] = views
    return targets


def write_gem_jsonl(rows: Sequence[GEMTokenizedRow], path: Path) -> None:
    """Atomically write plain JSONL accepted by official GEM's `load_dataset('json')`."""

    if not rows:
        raise ValueError("GEM training data cannot be empty")
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
                json.dump(row, handle, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_gem_rows(
    baseline: BaselineConfig,
    examples: Sequence[CanonicalExample],
    *,
    teacher_targets: TeacherTargets,
    tokenizer: Any,
    max_length: int,
) -> tuple[GEMTokenizedRow, ...]:
    """Build the pretokenized rows consumed unchanged by official GEM `train.py`."""

    if baseline.backend != "gem":
        raise ValueError(f"baseline {baseline.id} does not use GEM")
    if baseline.trajectory_source != "teacher" or baseline.target_view != "diverse4":
        raise ValueError(f"baseline {baseline.id} does not declare teacher diverse4 targets")

    rows: list[GEMTokenizedRow] = []
    for example in examples:
        try:
            targets = tuple(teacher_targets[example.id][baseline.target_view])
        except KeyError as error:
            raise ValueError(f"missing diverse4 teacher targets for {example.id}") from error
        if len(targets) != 4:
            raise ValueError(f"diverse4 requires 4 targets for {example.id}")
        rows.extend(
            tokenize_gem_example(
                example,
                target=target,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            for target in targets
        )
    return tuple(rows)


def tokenize_gem_example(
    example: CanonicalExample,
    *,
    target: str,
    tokenizer: Any,
    max_length: int,
) -> GEMTokenizedRow:
    """Build the `input_ids`/`labels` row required by official GEM `train.py`."""

    prompt_messages = [{"role": "user", "content": example.student_prompt}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": target.strip()},
    ]
    prompt_ids = list(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    input_ids = list(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full training sequence")
    if len(input_ids) > max_length:
        raise ValueError(
            f"tokenized example {example.id} has {len(input_ids)} tokens, above {max_length}"
        )
    if len(input_ids) == len(prompt_ids):
        raise ValueError(f"tokenized example {example.id} has no completion tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }


def _resolve_under(root: Path, path: Path) -> Path:
    root = root.resolve()
    candidate = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path must remain under scratch root {root}: {path}")
    return candidate
