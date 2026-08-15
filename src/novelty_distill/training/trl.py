"""Data/config adapter for official Hugging Face TRL trainers."""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from novelty_distill.config import BaselineConfig
from novelty_distill.data.semantic_seeds import (
    GaussianSeedSpec,
    condition_prompt,
    gaussian_seed_values,
)
from novelty_distill.data.teacher_views import derive_random_k_texts
from novelty_distill.data.tomato import CanonicalExample
from novelty_distill.data.training_rows import (
    ChatTrainingRow,
    PromptCompletionRow,
    to_chat_row,
    to_prompt_completion_row,
)
from novelty_distill.training.provenance import atomic_json, file_provenance
from novelty_distill.training.runtime import invocation_runtime

TRLTrainingRow = ChatTrainingRow | PromptCompletionRow
TeacherTargets = Mapping[str, Mapping[str, Sequence[str]]]


def diversity_aware_reverse_kl_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any | None = None,
    *,
    gamma: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> Any:
    """Compute the DRKL objective from Luong, Tran, and Chen (2026).

    The target/non-target decomposition is evaluated in float32 to keep the
    non-target mass stable for confident large-vocabulary predictions. The
    target index at each position is the observed completion token.
    """

    if gamma <= 0:
        raise ValueError("DRKL gamma must be positive")
    if temperature <= 0:
        raise ValueError("DRKL temperature must be positive")
    if reduction not in {"none", "batchmean", "sum", "mean"}:
        raise ValueError(f"unsupported DRKL reduction: {reduction}")
    if labels is None:
        raise ValueError("DRKL requires target-token labels")

    import torch
    import torch.nn.functional as functional

    student_log_probs = functional.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    teacher_log_probs = functional.log_softmax(
        teacher_logits.float() / temperature, dim=-1
    )
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0).unsqueeze(-1)
    log_q_target = student_log_probs.gather(-1, safe_labels).squeeze(-1)
    log_p_target = teacher_log_probs.gather(-1, safe_labels).squeeze(-1)

    # log(1 - exp(x)) for x <= 0, with stable branches around log(1/2).
    def log1mexp(value: Any) -> Any:
        value = value.clamp(max=-torch.finfo(value.dtype).eps)
        cutoff = -0.6931471805599453
        return torch.where(
            value < cutoff,
            torch.log1p(-torch.exp(value)),
            torch.log(-torch.expm1(value)),
        )

    log_q_non_target = log1mexp(log_q_target)
    log_p_non_target = log1mexp(log_p_target)
    q_target = log_q_target.exp()
    q_non_target = log_q_non_target.exp()

    target_rkl = q_target * (log_q_target - log_p_target)
    full_rkl = (
        student_log_probs.exp() * (student_log_probs - teacher_log_probs)
    ).sum(dim=-1)
    normalized_non_target_rkl = (
        (full_rkl - target_rkl) / q_non_target
        - log_q_non_target
        + log_p_non_target
    )
    binary_target_rkl = target_rkl + q_non_target * (
        log_q_non_target - log_p_non_target
    )
    token_loss = binary_target_rkl + gamma * normalized_non_target_rkl
    token_loss = token_loss[valid]
    if not token_loss.numel():
        raise ValueError("DRKL received no active completion labels")
    if reduction == "none":
        return token_loss
    if reduction == "sum":
        return token_loss.sum()
    return token_loss.mean()


class TRLRunSpec(BaseModel):
    """A bounded, reproducible run using an official TRL trainer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_id: str
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    teacher_model: str | None = None
    teacher_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    teacher_targets: Path | None = None
    input: Path
    output_dir: Path
    max_examples: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    save_steps: int | None = Field(default=None, gt=0)
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_length: int = Field(gt=0)
    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = Field(default=0.8, gt=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0, ge=0, le=1)
    student_thinking: bool = False
    attention_implementation: Literal["sdpa", "flash_attention_2"]
    gradient_checkpointing: bool
    use_peft: bool
    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    seed: int = Field(ge=0)
    input_seed: GaussianSeedSpec | None = None

    @model_validator(mode="after")
    def teacher_fields_are_paired(self) -> "TRLRunSpec":
        if (self.teacher_model is None) != (self.teacher_revision is None):
            raise ValueError("teacher_model and teacher_revision must be set together")
        if self.save_steps is not None and self.save_steps > self.max_steps:
            raise ValueError("save_steps cannot exceed max_steps")
        return self


def load_trl_run_spec(path: Path) -> TRLRunSpec:
    with path.open(encoding="utf-8") as handle:
        return TRLRunSpec.model_validate(yaml.safe_load(handle))


def override_trl_baseline(
    spec: TRLRunSpec, baseline_id: str | None, *, run_suffix: str = "smoke"
) -> TRLRunSpec:
    """Reuse one backend config while keeping each staged output isolated."""

    if baseline_id is None:
        return spec
    cleaned = baseline_id.strip()
    if not cleaned or "/" in cleaned or ".." in cleaned:
        raise ValueError("baseline override must be a safe non-empty ID")
    suffix = run_suffix.strip()
    if not suffix or "/" in suffix or ".." in suffix:
        raise ValueError("run suffix must be a safe non-empty name")
    return spec.model_copy(
        update={
            "baseline_id": cleaned,
            "output_dir": Path("checkpoints") / f"{cleaned}-{suffix}",
        }
    )


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
    selection_seed: int = 17,
    input_seed: GaussianSeedSpec | None = None,
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
        if baseline.trajectory_source == "student":
            prompt = _condition_training_prompt(
                example.student_prompt,
                prompt_id=example.id,
                sample_index=0,
                selection_seed=selection_seed,
                input_seed=input_seed,
            )
            rows.append(
                {
                    "id": example.id,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ""},
                    ],
                }
            )
            continue
        if baseline.trajectory_source == "human":
            targets = (example.human_target,)
        else:
            views = teacher_targets.get(example.id)
            if views is None:
                raise ValueError(f"missing teacher targets for {example.id}")
            if baseline.target_view == "random4":
                try:
                    targets = derive_random_k_texts(
                        views["all8"],
                        prompt_id=example.id,
                        seed=selection_seed,
                        k=4,
                    )
                except KeyError as error:
                    raise ValueError(
                        f"missing all8 teacher targets for random4/{example.id}"
                    ) from error
            else:
                try:
                    targets = tuple(views[baseline.target_view])
                except KeyError as error:
                    raise ValueError(
                        f"missing {baseline.target_view} teacher targets for {example.id}"
                    ) from error
            expected = 4 if baseline.target_view in {"random4", "diverse4"} else 1
            if len(targets) != expected:
                raise ValueError(
                    f"{baseline.target_view} requires {expected} targets for {example.id}"
                )
        for target_index, target in enumerate(targets):
            prompt = _condition_training_prompt(
                example.student_prompt,
                prompt_id=example.id,
                sample_index=target_index,
                selection_seed=selection_seed,
                input_seed=input_seed,
            )
            if baseline.backend == "trl_sft":
                row = to_prompt_completion_row(example, target=target)
                row["prompt"][0]["content"] = prompt
                rows.append(row)
            else:
                row = to_chat_row(example, target=target)
                row["messages"][0]["content"] = prompt
                rows.append(row)
    return tuple(rows)


def _condition_training_prompt(
    prompt: str,
    *,
    prompt_id: str,
    sample_index: int,
    selection_seed: int,
    input_seed: GaussianSeedSpec | None,
) -> str:
    if input_seed is None:
        return prompt
    values = gaussian_seed_values(
        prompt_id=prompt_id,
        sample_index=sample_index,
        generation_seed=selection_seed,
        spec=input_seed,
    )
    return condition_prompt(prompt, values=values)


def encode_prompt_preserving_chatml_example(
    example: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
    enable_thinking: bool = False,
) -> dict[str, list[int] | int]:
    """Encode ChatML while preserving the full prompt and completion prefix."""

    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("GKD examples require at least user and assistant messages")
    formatted_prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    formatted_message = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    prompt_ids = list(
        tokenizer(
            formatted_prompt,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
    )
    message_ids = list(
        tokenizer(
            formatted_message,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
    )
    if message_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("full ChatML message does not preserve the rendered prompt prefix")
    if len(prompt_ids) >= max_length:
        raise ValueError(
            f"GKD prompt length {len(prompt_ids)} leaves no completion budget at {max_length}"
        )
    completion_ids = message_ids[len(prompt_ids) :]
    completion_budget = max_length - len(prompt_ids)
    kept_completion = completion_ids[:completion_budget]
    if not kept_completion:
        raise ValueError("GKD example has no assistant completion tokens")
    input_ids = [*prompt_ids, *kept_completion]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + kept_completion,
        "prompt_ids": prompt_ids,
        "prompt_attention_mask": [1] * len(prompt_ids),
        "truncated_completion_tokens": len(completion_ids) - len(kept_completion),
    }


class PromptPreservingChatMLCollator:
    """Official-GKD tensor schema with an explicit prompt-preserving truncation policy."""

    def __init__(
        self, tokenizer: Any, *, max_length: int, enable_thinking: bool = False
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = tuple(
            encode_prompt_preserving_chatml_example(
                example,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                enable_thinking=self.enable_thinking,
            )
            for example in examples
        )
        return {
            "input_ids": _left_pad(
                tuple(item["input_ids"] for item in encoded),
                value=self.tokenizer.pad_token_id,
                torch=torch,
            ),
            "attention_mask": _left_pad(
                tuple(item["attention_mask"] for item in encoded), value=0, torch=torch
            ),
            "labels": _left_pad(
                tuple(item["labels"] for item in encoded), value=-100, torch=torch
            ),
            "prompts": _left_pad(
                tuple(item["prompt_ids"] for item in encoded),
                value=self.tokenizer.pad_token_id,
                torch=torch,
            ),
            "prompt_attention_mask": _left_pad(
                tuple(item["prompt_attention_mask"] for item in encoded),
                value=0,
                torch=torch,
            ),
        }


def configure_gkd_generation(
    trainer: Any, spec: TRLRunSpec
) -> dict[str, float | int | bool]:
    """Pin official GKD on-policy generation to the frozen sampling contract."""

    controls: dict[str, float | int | bool] = {
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "min_p": spec.min_p,
        "max_new_tokens": spec.max_new_tokens,
        "student_thinking": spec.student_thinking,
    }
    generation_config = trainer.generation_config
    for name in ("temperature", "top_p", "top_k", "min_p", "max_new_tokens"):
        setattr(generation_config, name, controls[name])
    return controls


def _left_pad(sequences: Sequence[Any], *, value: int, torch: Any) -> Any:
    if not sequences:
        raise ValueError("cannot pad an empty GKD batch")
    width = max(len(sequence) for sequence in sequences)
    output = torch.full((len(sequences), width), value, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        if sequence:
            output[index, -len(sequence) :] = torch.tensor(sequence, dtype=torch.long)
    return output


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
    training_artifacts = {
        "input": file_provenance(input_path),
        "registry": file_provenance(registry_path),
    }
    if baseline.trajectory_source == "teacher" and spec.teacher_targets is not None:
        training_artifacts["teacher_targets"] = file_provenance(
            _resolve_under(scratch_root, spec.teacher_targets)
        )
    examples = load_canonical_examples(input_path, limit=spec.max_examples)
    rows = build_trl_rows(
        baseline,
        examples,
        selection_seed=spec.seed,
        input_seed=spec.input_seed,
        teacher_targets=(
            teacher_targets
            if teacher_targets is not None
            else _load_configured_teacher_targets(
                spec,
                scratch_root,
                required=baseline.trajectory_source == "teacher",
            )
        ),
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
        "save_steps": spec.save_steps or spec.max_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": spec.seed,
        "input_seed": (
            spec.input_seed.model_dump(mode="json") if spec.input_seed is not None else None
        ),
        "data_seed": spec.seed,
    }
    gkd_context_audit: dict[str, int | float] | None = None
    gkd_generation_controls: dict[str, float | int | bool] | None = None
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
        encoded_audit = tuple(
            encode_prompt_preserving_chatml_example(
                row,
                tokenizer=tokenizer,
                max_length=spec.max_length,
                enable_thinking=spec.student_thinking,
            )
            for row in rows
        )
        truncated_tokens = tuple(
            int(item["truncated_completion_tokens"]) for item in encoded_audit
        )
        gkd_context_audit = {
            "rows": len(rows),
            "truncated_rows": sum(value > 0 for value in truncated_tokens),
            "truncated_row_rate": sum(value > 0 for value in truncated_tokens) / len(rows),
            "truncated_completion_tokens": sum(truncated_tokens),
        }
        trainer_type = GKDTrainer
        if baseline.divergence == "diversity_aware_reverse_kl":
            drkl_gamma = baseline.drkl_gamma
            if drkl_gamma is None:  # guarded by BaselineConfig; keeps runtime fail-closed
                raise ValueError("DRKL baseline is missing drkl_gamma")

            class DiversityAwareGKDTrainer(GKDTrainer):
                def generalized_jsd_loss(
                    self,
                    student_logits: Any,
                    teacher_logits: Any,
                    labels: Any | None = None,
                    beta: float = 0.5,
                    temperature: float = 1.0,
                    reduction: str = "batchmean",
                ) -> Any:
                    del beta
                    return diversity_aware_reverse_kl_loss(
                        student_logits,
                        teacher_logits,
                        labels,
                        gamma=drkl_gamma,
                        temperature=temperature,
                        reduction=reduction,
                    )

            trainer_type = DiversityAwareGKDTrainer

        trainer = trainer_type(
            model=student_model,
            teacher_model=teacher_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=PromptPreservingChatMLCollator(
                tokenizer,
                max_length=spec.max_length,
                enable_thinking=spec.student_thinking,
            ),
        )
        gkd_generation_controls = configure_gkd_generation(trainer, spec)
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
        "model": spec.model,
        "revision": spec.revision,
        "teacher_model": spec.teacher_model,
        "teacher_revision": spec.teacher_revision,
        "student_thinking": spec.student_thinking,
        "lmbda": baseline.lmbda,
        "beta": baseline.beta,
        "divergence": baseline.divergence,
        "drkl_gamma": baseline.drkl_gamma,
        "trajectory_source": baseline.trajectory_source,
        "target_view": baseline.target_view,
        "dataset_revision": examples[0].dataset_revision,
        "example_ids": [example.id for example in examples],
        "training_artifacts": training_artifacts,
        "training_rows": len(rows),
        "optimizer_example_exposures": (
            spec.max_steps
            * spec.per_device_train_batch_size
            * spec.gradient_accumulation_steps
        ),
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps or spec.max_steps,
        "seed": spec.seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": os.environ.get("NOVELTY_GIT_COMMIT"),
        "metrics": train_result.metrics,
        "invocation_runtime": invocation,
        "gkd_context_audit": gkd_context_audit,
        "gkd_generation_controls": gkd_generation_controls,
        "final_dir": str(final_dir),
    }
    metadata_path = output_dir / "run_metadata.json"
    atomic_json(metadata_path, metadata)
    return metadata


def _resolve_under(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path must stay under scratch root {root}: {path}")
    return resolved


def _load_configured_teacher_targets(
    spec: TRLRunSpec, scratch_root: Path, *, required: bool
) -> TeacherTargets:
    if not required:
        return {}
    if spec.teacher_targets is None:
        raise ValueError(f"baseline {spec.baseline_id} requires a teacher-target artifact")
    from novelty_distill.training.gem import load_teacher_targets

    return load_teacher_targets(_resolve_under(scratch_root, spec.teacher_targets))
