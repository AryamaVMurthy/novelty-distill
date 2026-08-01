"""Requests for the official SGLang OpenAI-compatible server."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GenerationSpec(BaseModel):
    """Sampling controls that must be identical across comparable models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    revision: str = Field(min_length=40, max_length=40)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_new_tokens: int = Field(gt=0)
    samples_per_prompt: int = Field(gt=0)
    seed: int = Field(ge=0)
    enable_thinking: bool = False


class Prompt(BaseModel):
    """A stable prompt identifier and its rendered student-visible text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class GenerationRecord(BaseModel):
    """One auditable completion returned by SGLang."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    sample_index: int = Field(ge=0)
    text: str
    finish_reason: str | None
    model: str
    request_id: str
    config_hash: str
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


def build_chat_completion_payload(prompt: str, spec: GenerationSpec) -> dict[str, Any]:
    """Build one SGLang `/v1/chat/completions` request."""

    return {
        "model": spec.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "max_tokens": spec.max_new_tokens,
        "n": spec.samples_per_prompt,
        "seed": spec.seed,
        "chat_template_kwargs": {"enable_thinking": spec.enable_thinking},
    }


def generation_fingerprint(spec: GenerationSpec) -> str:
    """Return a stable hash of every generation control."""

    encoded = json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_chat_completion_response(
    prompt_id: str, spec: GenerationSpec, response: Mapping[str, Any]
) -> tuple[GenerationRecord, ...]:
    """Validate and normalize one OpenAI-compatible SGLang response."""

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != spec.samples_per_prompt:
        raise ValueError(
            f"expected {spec.samples_per_prompt} choices, received "
            f"{len(choices) if isinstance(choices, list) else 'an invalid value'}"
        )

    request_id = str(response.get("id", "")).strip()
    if not request_id:
        raise ValueError("SGLang response has no request id")
    response_model = str(response.get("model", spec.model))
    usage = response.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}

    records: list[GenerationRecord] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            raise ValueError("SGLang response contains a non-object choice")
        message = choice.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise ValueError("SGLang response choice has no text content")
        records.append(
            GenerationRecord(
                prompt_id=prompt_id,
                sample_index=int(choice["index"]),
                text=message["content"],
                finish_reason=(
                    str(choice["finish_reason"])
                    if choice.get("finish_reason") is not None
                    else None
                ),
                model=response_model,
                request_id=request_id,
                config_hash=generation_fingerprint(spec),
                prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                completion_tokens=_optional_int(usage.get("completion_tokens")),
            )
        )

    records.sort(key=lambda record: record.sample_index)
    expected_indices = list(range(spec.samples_per_prompt))
    if [record.sample_index for record in records] != expected_indices:
        raise ValueError(f"choice indices must be exactly {expected_indices}")
    return tuple(records)


def prompt_shard_path(output_dir: Path, prompt_id: str) -> Path:
    """Map an arbitrary prompt id to a filesystem-safe shard name."""

    digest = hashlib.sha256(prompt_id.encode()).hexdigest()
    return output_dir / f"{digest}.json"


def write_prompt_shard(
    output_dir: Path,
    records: Iterable[GenerationRecord],
    spec: GenerationSpec,
) -> Path:
    """Atomically persist one complete prompt, which is the unit of resume."""

    record_tuple = tuple(records)
    _validate_records(record_tuple, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = prompt_shard_path(output_dir, record_tuple[0].prompt_id)
    payload = {
        "schema_version": 1,
        "prompt_id": record_tuple[0].prompt_id,
        "config_hash": generation_fingerprint(spec),
        "records": [record.model_dump(mode="json") for record in record_tuple],
    }

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_prompt_shard(path: Path, spec: GenerationSpec) -> tuple[GenerationRecord, ...]:
    """Load a shard and reject stale, partial, or incompatible output."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported generation shard schema in {path}")
    if payload.get("config_hash") != generation_fingerprint(spec):
        raise ValueError(f"generation settings changed for existing shard {path}")
    records = tuple(GenerationRecord.model_validate(item) for item in payload.get("records", ()))
    _validate_records(records, spec)
    if payload.get("prompt_id") != records[0].prompt_id:
        raise ValueError(f"prompt id mismatch in {path}")
    return records


def pending_prompts(
    prompts: Iterable[Prompt], output_dir: Path, spec: GenerationSpec
) -> tuple[Prompt, ...]:
    """Return only prompts without a complete shard for this exact configuration."""

    pending: list[Prompt] = []
    for prompt in prompts:
        shard = prompt_shard_path(output_dir, prompt.id)
        if not shard.exists():
            pending.append(prompt)
            continue
        records = load_prompt_shard(shard, spec)
        if records[0].prompt_id != prompt.id:
            raise ValueError(f"prompt hash collision at {shard}")
    return tuple(pending)


PostJSON = Callable[[str, dict[str, Any], float], Mapping[str, Any]]


def generate_prompt(
    prompt: Prompt,
    spec: GenerationSpec,
    output_dir: Path,
    *,
    base_url: str,
    timeout: float = 600,
    post: PostJSON | None = None,
) -> tuple[GenerationRecord, ...]:
    """Generate one complete prompt shard, or validate and reuse it on resume."""

    shard = prompt_shard_path(output_dir, prompt.id)
    if shard.exists():
        records = load_prompt_shard(shard, spec)
        if records[0].prompt_id != prompt.id:
            raise ValueError(f"prompt hash collision at {shard}")
        return records

    post_json = post or _httpx_post
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    response = post_json(endpoint, build_chat_completion_payload(prompt.text, spec), timeout)
    records = parse_chat_completion_response(prompt.id, spec, response)
    write_prompt_shard(output_dir, records, spec)
    return records


def _validate_records(records: tuple[GenerationRecord, ...], spec: GenerationSpec) -> None:
    if len(records) != spec.samples_per_prompt:
        raise ValueError(f"expected exactly {spec.samples_per_prompt} records")
    prompt_ids = {record.prompt_id for record in records}
    if len(prompt_ids) != 1:
        raise ValueError("a shard must contain exactly one prompt id")
    expected_hash = generation_fingerprint(spec)
    if any(record.config_hash != expected_hash for record in records):
        raise ValueError("record generation settings do not match the requested settings")
    if [record.sample_index for record in records] != list(range(spec.samples_per_prompt)):
        raise ValueError("records must be ordered and have contiguous sample indices")


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _httpx_post(url: str, payload: dict[str, Any], timeout: float) -> Mapping[str, Any]:
    try:
        import httpx
    except ImportError as error:
        raise RuntimeError("install the pinned inference environment to call SGLang") from error

    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, Mapping):
        raise ValueError("SGLang returned a non-object JSON response")
    return body
