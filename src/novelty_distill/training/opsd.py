"""Data/config adapter for the pinned official OPSD trainer."""

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from novelty_distill.config import BaselineConfig
from novelty_distill.data.tomato import CanonicalExample
from novelty_distill.data.training_rows import OPSDTrainingRow, to_opsd_row
from novelty_distill.official import checkout_official_repository, load_official_repositories
from novelty_distill.training.provenance import atomic_json, file_provenance
from novelty_distill.training.runtime import invocation_runtime
from novelty_distill.training.trl import load_canonical_examples


class OPSDRunSpec(BaseModel):
    """A bounded, reproducible run through the official OPSD trainer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_id: str
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    input: Path
    output_dir: Path
    max_examples: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    save_steps: int | None = Field(default=None, gt=0)
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_length: int = Field(gt=0)
    max_completion_length: int = Field(gt=0)
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    attention_implementation: Literal["sdpa", "flash_attention_2"]
    gradient_checkpointing: bool
    use_peft: bool
    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    jsd_token_clip: float | None = Field(default=None, gt=0)
    student_thinking: bool
    teacher_thinking: bool
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def save_cadence_is_bounded(self) -> "OPSDRunSpec":
        if self.save_steps is not None and self.save_steps > self.max_steps:
            raise ValueError("save_steps cannot exceed max_steps")
        if self.jsd_token_clip is not None:
            raise ValueError(
                "OPSD token clipping acts on signed vocabulary contributions before their "
                "divergence reduction; use jsd_token_clip: null"
            )
        return self


def load_opsd_run_spec(path: Path) -> OPSDRunSpec:
    with path.open(encoding="utf-8") as handle:
        return OPSDRunSpec.model_validate(yaml.safe_load(handle))


def override_opsd_baseline(
    spec: OPSDRunSpec, baseline_id: str | None, *, run_suffix: str = "smoke"
) -> OPSDRunSpec:
    """Reuse the OPSD runtime controls for staged E2/E3/E4 runs."""

    if baseline_id is None:
        return spec
    cleaned = baseline_id.strip()
    if cleaned not in {"E2", "E3", "E4"}:
        raise ValueError("OPSD baseline override must be E2, E3, or E4")
    suffix = run_suffix.strip()
    if not suffix or "/" in suffix or ".." in suffix:
        raise ValueError("run suffix must be a safe non-empty name")
    return spec.model_copy(
        update={
            "baseline_id": cleaned,
            "output_dir": Path("checkpoints") / f"{cleaned}-{suffix}",
        }
    )


def build_opsd_rows(
    baseline: BaselineConfig, examples: Sequence[CanonicalExample]
) -> tuple[OPSDTrainingRow, ...]:
    """Build the `problem`/`solution` rows consumed by the official collator."""

    if baseline.backend != "opsd":
        raise ValueError(f"baseline {baseline.id} does not use OPSD")
    if baseline.teacher_context == "ordinary":
        return tuple(
            {"id": example.id, "problem": example.student_prompt, "solution": ""}
            for example in examples
        )
    return tuple(to_opsd_row(example) for example in examples)


def render_matched_prompt_pairs(
    features: Sequence[Mapping[str, str]],
    *,
    tokenizer: Any,
    enable_thinking: bool,
) -> tuple[tuple[str, str], ...]:
    """Render byte-identical ordinary contexts for the E4 negative control."""

    pairs: list[tuple[str, str]] = []
    for feature in features:
        messages = [{"role": "user", "content": feature["problem"]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        pairs.append((prompt, prompt))
    return tuple(pairs)


def render_privileged_prompt_pairs(
    features: Sequence[Mapping[str, str]],
    *,
    tokenizer: Any,
    student_thinking: bool,
    teacher_thinking: bool,
) -> tuple[tuple[str, str], ...]:
    """Render one unchanged student task and one teacher-only privileged extension."""

    pairs: list[tuple[str, str]] = []
    for feature in features:
        problem = str(feature["problem"]).strip()
        solution = str(feature["solution"]).strip()
        if not problem or not solution:
            raise ValueError("privileged OPSD rows require non-empty problem and context")
        student_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=student_thinking,
        )
        teacher_content = (
            f"{problem}\n\n"
            "Private context available only to the teacher:\n"
            f"{solution}\n\n"
            "Use this private context as evidence when answering the original request. "
            "Do not mention the private context or the fact that it was supplied."
        )
        teacher_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": teacher_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=teacher_thinking,
        )
        pairs.append((student_prompt, teacher_prompt))
    return tuple(pairs)


def opsd_dataset_kwargs() -> dict[str, bool]:
    """Keep the raw columns required by the official OPSD data collator."""

    return {"skip_prepare_dataset": True}


class MatchedContextCollator:
    """OPSD collator for the no-privilege control with identical contexts."""

    def __init__(self, tokenizer: Any, *, max_length: int, enable_thinking: bool):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking

    def __call__(self, features: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        import torch

        pairs = render_matched_prompt_pairs(
            features,
            tokenizer=self.tokenizer,
            enable_thinking=self.enable_thinking,
        )
        prompts = [student for student, _teacher in pairs]
        unpadded = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        lengths = [len(input_ids) for input_ids in unpadded["input_ids"]]
        batch_length = max(lengths)
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=batch_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        lengths_tensor = torch.tensor(lengths)
        return {
            "student_prompts": input_ids,
            "student_prompt_attention_mask": attention_mask,
            "student_prompt_length": batch_length,
            "student_prompt_lengths_per_example": lengths_tensor,
            "teacher_prompts": input_ids.clone(),
            "teacher_prompt_attention_mask": attention_mask.clone(),
            "teacher_prompt_length": batch_length,
            "teacher_prompt_lengths_per_example": lengths_tensor.clone(),
        }


class PrivilegedContextCollator:
    """Task-faithful TOMATO adapter for the official fixed-teacher OPSD trainer."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        student_thinking: bool,
        teacher_thinking: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking

    def __call__(self, features: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        pairs = render_privileged_prompt_pairs(
            features,
            tokenizer=self.tokenizer,
            student_thinking=self.student_thinking,
            teacher_thinking=self.teacher_thinking,
        )
        student = _encode_prompt_batch(
            tuple(prompt for prompt, _teacher in pairs),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        teacher = _encode_prompt_batch(
            tuple(prompt for _student, prompt in pairs),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return {
            "student_prompts": student["input_ids"],
            "student_prompt_attention_mask": student["attention_mask"],
            "student_prompt_length": student["batch_length"],
            "student_prompt_lengths_per_example": student["lengths"],
            "teacher_prompts": teacher["input_ids"],
            "teacher_prompt_attention_mask": teacher["attention_mask"],
            "teacher_prompt_length": teacher["batch_length"],
            "teacher_prompt_lengths_per_example": teacher["lengths"],
        }


def _encode_prompt_batch(
    prompts: Sequence[str], *, tokenizer: Any, max_length: int
) -> dict[str, Any]:
    import torch

    unpadded = tokenizer(
        list(prompts),
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    lengths = [len(input_ids) for input_ids in unpadded["input_ids"]]
    if not lengths:
        raise ValueError("OPSD collator requires a non-empty batch")
    batch_length = max(lengths)
    encoded = tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=batch_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "batch_length": batch_length,
        "lengths": torch.tensor(lengths),
    }


def execute_opsd_training(
    spec: OPSDRunSpec,
    *,
    scratch_root: Path,
    registry_path: Path,
    manifest_path: Path,
    official_root: Path,
) -> dict[str, object]:
    """Execute the pinned author OPSD trainer without copying its implementation."""

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl.experimental.gold import GOLDConfig

    from novelty_distill.config import load_baseline_registry

    registry = load_baseline_registry(registry_path)
    matches = [baseline for baseline in registry.baselines if baseline.id == spec.baseline_id]
    if len(matches) != 1:
        raise ValueError(f"baseline {spec.baseline_id} is not uniquely defined")
    baseline = matches[0]
    if baseline.backend != "opsd":
        raise ValueError(f"baseline {baseline.id} does not use OPSD")
    if baseline.lmbda != 1.0:
        raise NotImplementedError(
            "the pinned OPSD trainer is hard-coded on-policy; E1 needs a reviewed upstream patch"
        )
    if not spec.use_peft:
        raise ValueError("fixed-teacher OPSD requires PEFT so the base checkpoint stays frozen")

    repositories = load_official_repositories(manifest_path)
    checkout = checkout_official_repository(repositories["opsd"], official_root)
    sys.path.insert(0, str(checkout))
    from opsd_trainer import OPSDTrainer

    input_path = _resolve_under(scratch_root, spec.input)
    output_dir = _resolve_under(scratch_root, spec.output_dir)
    examples = load_canonical_examples(input_path, limit=spec.max_examples)
    rows = build_opsd_rows(baseline, examples)
    train_dataset = Dataset.from_list(list(rows))

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model,
        revision=spec.revision,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = GOLDConfig(
        output_dir=str(output_dir),
        max_steps=spec.max_steps,
        per_device_train_batch_size=spec.per_device_train_batch_size,
        gradient_accumulation_steps=spec.gradient_accumulation_steps,
        learning_rate=spec.learning_rate,
        max_length=spec.max_length,
        max_completion_length=spec.max_completion_length,
        temperature=spec.temperature,
        top_p=spec.top_p,
        lmbda=baseline.lmbda,
        beta=baseline.beta,
        seq_kd=False,
        gradient_checkpointing=spec.gradient_checkpointing,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=spec.save_steps or spec.max_steps,
        save_total_limit=2,
        report_to="none",
        seed=spec.seed,
        data_seed=spec.seed,
        use_vllm=False,
        model_init_kwargs={
            "revision": spec.revision,
            "torch_dtype": "bfloat16",
            "attn_implementation": spec.attention_implementation,
            "use_cache": not spec.gradient_checkpointing,
        },
        dataset_kwargs=opsd_dataset_kwargs(),
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
    )
    if baseline.teacher_context == "ordinary":
        data_collator = MatchedContextCollator(
            tokenizer,
            max_length=spec.max_length,
            enable_thinking=spec.student_thinking,
        )
    else:
        data_collator = PrivilegedContextCollator(
            tokenizer,
            max_length=spec.max_length,
            student_thinking=spec.student_thinking,
            teacher_thinking=spec.teacher_thinking,
        )
    trainer = OPSDTrainer(
        model=spec.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        fixed_teacher=True,
        reason_first=False,
        top_k_loss=None,
        jsd_token_clip=spec.jsd_token_clip,
        use_ema_teacher=False,
        student_thinking=spec.student_thinking,
        teacher_thinking=spec.teacher_thinking,
    )
    last_checkpoint = get_last_checkpoint(str(output_dir))
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    invocation = invocation_runtime(
        last_checkpoint=last_checkpoint,
        end_step=int(trainer.state.global_step),
        metrics=train_result.metrics,
    )
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    metadata: dict[str, object] = {
        "baseline_id": baseline.id,
        "backend": baseline.backend,
        "run_spec": spec.model_dump(mode="json"),
        "official_commit": repositories["opsd"].commit,
        "model": spec.model,
        "revision": spec.revision,
        "lmbda": baseline.lmbda,
        "beta": baseline.beta,
        "divergence": baseline.divergence,
        "trajectory_source": baseline.trajectory_source,
        "target_view": baseline.target_view,
        "teacher_context": baseline.teacher_context,
        "data_collator": (
            "tomato_matched_context"
            if baseline.teacher_context == "ordinary"
            else "tomato_privileged_context"
        ),
        "student_prompt_contract": "ordinary_tomato_chat_prompt",
        "dataset_revision": examples[0].dataset_revision,
        "example_ids": [example.id for example in examples],
        "training_artifacts": {
            "input": file_provenance(input_path),
            "registry": file_provenance(registry_path),
            "official_manifest": file_provenance(manifest_path),
        },
        "training_rows": len(rows),
        "optimizer_example_exposures": (
            spec.max_steps * spec.per_device_train_batch_size * spec.gradient_accumulation_steps
        ),
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps or spec.max_steps,
        "seed": spec.seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": os.environ.get("NOVELTY_GIT_COMMIT"),
        "metrics": train_result.metrics,
        "invocation_runtime": invocation,
        "final_dir": str(final_dir),
    }
    atomic_json(output_dir / "run_metadata.json", metadata)
    return metadata


def _resolve_under(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path must stay under scratch root {root}: {path}")
    return resolved
