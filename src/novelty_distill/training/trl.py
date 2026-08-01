"""Data/config adapter for official Hugging Face TRL trainers."""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from novelty_distill.config import BaselineConfig
from novelty_distill.data.tomato import CanonicalExample
from novelty_distill.data.training_rows import (
    ChatTrainingRow,
    PromptCompletionRow,
    to_chat_row,
    to_prompt_completion_row,
)

TRLTrainingRow = ChatTrainingRow | PromptCompletionRow
TeacherTargets = Mapping[str, Mapping[str, Sequence[str]]]


class TRLRunSpec(BaseModel):
    """A bounded, reproducible run using an official TRL trainer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_id: str
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    teacher_model: str | None = None
    teacher_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    input: Path
    output_dir: Path
    max_examples: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_length: int = Field(gt=0)
    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = Field(default=0.8, gt=0)
    attention_implementation: Literal["sdpa", "flash_attention_2"]
    gradient_checkpointing: bool
    use_peft: bool
    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def teacher_fields_are_paired(self) -> "TRLRunSpec":
        if (self.teacher_model is None) != (self.teacher_revision is None):
            raise ValueError("teacher_model and teacher_revision must be set together")
        return self


def load_trl_run_spec(path: Path) -> TRLRunSpec:
    with path.open(encoding="utf-8") as handle:
        return TRLRunSpec.model_validate(yaml.safe_load(handle))


def load_canonical_examples(path: Path, *, limit: int) -> tuple[CanonicalExample, ...]:
    """Load a bounded canonical JSONL prefix for a reproducible training run."""

    if limit <= 0:
        raise ValueError("example limit must be positive")
    examples: list[CanonicalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(CanonicalExample.model_validate_json(line))
            if len(examples) == limit:
                break
    if len(examples) < limit:
        raise ValueError(f"requested {limit} examples but {path} contains {len(examples)}")
    return tuple(examples)


def build_trl_rows(
    baseline: BaselineConfig,
    examples: Sequence[CanonicalExample],
    *,
    teacher_targets: TeacherTargets,
) -> tuple[TRLTrainingRow, ...]:
    """Build rows for an official TRL backend without changing target semantics."""

    if baseline.backend not in {"trl_sft", "trl_gkd"}:
        raise ValueError(f"baseline {baseline.id} does not use TRL")
    if baseline.trajectory_source not in {"human", "teacher", "student"}:
        raise NotImplementedError(
            f"target source {baseline.trajectory_source} is not yet wired for {baseline.id}"
        )

    rows: list[TRLTrainingRow] = []
    for example in examples:
        if baseline.trajectory_source in {"human", "student"}:
            targets = (example.human_target,)
        else:
            try:
                targets = tuple(teacher_targets[example.id][baseline.target_view])
            except KeyError as error:
                raise ValueError(
                    f"missing {baseline.target_view} teacher targets for {example.id}"
                ) from error
            expected = 4 if baseline.target_view == "diverse4" else 1
            if len(targets) != expected:
                raise ValueError(
                    f"{baseline.target_view} requires {expected} targets for {example.id}"
                )
        for target in targets:
            if baseline.backend == "trl_sft":
                rows.append(to_prompt_completion_row(example, target=target))
            else:
                rows.append(to_chat_row(example, target=target))
    return tuple(rows)


def execute_trl_training(
    spec: TRLRunSpec,
    *,
    scratch_root: Path,
    registry_path: Path,
    teacher_targets: TeacherTargets | None = None,
) -> dict[str, object]:
    """Execute a bounded run through the official TRL trainer API."""

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint

    from novelty_distill.config import load_baseline_registry

    registry = load_baseline_registry(registry_path)
    matches = [baseline for baseline in registry.baselines if baseline.id == spec.baseline_id]
    if len(matches) != 1:
        raise ValueError(f"baseline {spec.baseline_id} is not uniquely defined")
    baseline = matches[0]
    if baseline.backend not in {"trl_sft", "trl_gkd"}:
        raise NotImplementedError(
            f"executable support for official backend {baseline.backend} is not ready"
        )
    if baseline.backend == "trl_gkd" and (
        spec.teacher_model is None or spec.teacher_revision is None
    ):
        raise ValueError(f"baseline {baseline.id} requires a pinned teacher model")

    input_path = _resolve_under(scratch_root, spec.input)
    output_dir = _resolve_under(scratch_root, spec.output_dir)
    examples = load_canonical_examples(input_path, limit=spec.max_examples)
    rows = build_trl_rows(
        baseline,
        examples,
        teacher_targets=teacher_targets or {},
    )
    train_dataset = Dataset.from_list(list(rows))

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model,
        revision=spec.revision,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir.mkdir(parents=True, exist_ok=True)
    peft_config = None
    if spec.use_peft:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=spec.lora_r,
            lora_alpha=spec.lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
        )
    common_training_args = {
        "output_dir": str(output_dir),
        "max_steps": spec.max_steps,
        "per_device_train_batch_size": spec.per_device_train_batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "learning_rate": spec.learning_rate,
        "max_length": spec.max_length,
        "gradient_checkpointing": spec.gradient_checkpointing,
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": spec.max_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": spec.seed,
        "data_seed": spec.seed,
    }
    if baseline.backend == "trl_sft":
        from trl import SFTConfig, SFTTrainer

        training_args = SFTConfig(
            **common_training_args,
            completion_only_loss=True,
            model_init_kwargs={
                "revision": spec.revision,
                "torch_dtype": "bfloat16",
                "attn_implementation": spec.attention_implementation,
                "use_cache": not spec.gradient_checkpointing,
            },
        )
        trainer = SFTTrainer(
            model=spec.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        import torch
        from trl.experimental.gkd import GKDConfig, GKDTrainer

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": spec.attention_implementation,
            "use_cache": not spec.gradient_checkpointing,
            "low_cpu_mem_usage": True,
        }
        student_model = AutoModelForCausalLM.from_pretrained(
            spec.model,
            revision=spec.revision,
            **model_kwargs,
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            spec.teacher_model,
            revision=spec.teacher_revision,
            **model_kwargs,
        )
        training_args = GKDConfig(
            **common_training_args,
            lmbda=baseline.lmbda,
            beta=baseline.beta,
            temperature=spec.temperature,
            max_new_tokens=spec.max_new_tokens,
            seq_kd=False,
        )
        trainer = GKDTrainer(
            model=student_model,
            teacher_model=teacher_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        if os.environ.get("NOVELTY_GKD_DIAGNOSTIC") == "1":
            probe = trainer.data_collator([rows[0]])
            print(
                json.dumps(
                    {
                        "gkd_probe_shapes": {
                            key: list(value.shape)
                            for key, value in probe.items()
                            if hasattr(value, "shape")
                        },
                        "prompt_active_tokens": int(probe["prompt_attention_mask"].sum()),
                        "completion_active_tokens": int((probe["labels"] != -100).sum()),
                    },
                    sort_keys=True,
                )
            )
    last_checkpoint = get_last_checkpoint(str(output_dir))
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    metadata: dict[str, object] = {
        "baseline_id": baseline.id,
        "backend": baseline.backend,
        "model": spec.model,
        "revision": spec.revision,
        "teacher_model": spec.teacher_model,
        "teacher_revision": spec.teacher_revision,
        "lmbda": baseline.lmbda,
        "beta": baseline.beta,
        "trajectory_source": baseline.trajectory_source,
        "target_view": baseline.target_view,
        "dataset_revision": examples[0].dataset_revision,
        "example_ids": [example.id for example in examples],
        "max_steps": spec.max_steps,
        "seed": spec.seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": os.environ.get("NOVELTY_GIT_COMMIT"),
        "metrics": train_result.metrics,
        "final_dir": str(final_dir),
    }
    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def _resolve_under(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path must stay under scratch root {root}: {path}")
    return resolved
