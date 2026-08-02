"""Requests for the official SGLang OpenAI-compatible server."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SAMPLING_STRATEGY = "single-request-per-sample-v1"
INFERENCE_ARTIFACT_PATTERNS = (
    "config.json",
    "generation_config.json",
    "adapter_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)


class GenerationSpec(BaseModel):
    """Sampling controls that must be identical across comparable models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    revision: str = Field(min_length=40, max_length=40)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(default=20, gt=0)
    min_p: float = Field(default=0, ge=0, le=1)
    max_new_tokens: int = Field(gt=0)
    samples_per_prompt: int = Field(gt=0)
    seed: int = Field(ge=0)
    enable_thinking: bool = False
    response_instruction: str | None = Field(default=None, min_length=1)
    lora_path: str | None = Field(default=None, min_length=1)


class Prompt(BaseModel):
    """A stable prompt identifier and its rendered student-visible text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


def load_prompts(path: Path) -> tuple[Prompt, ...]:
    """Load canonical JSONL prompts and reject malformed or duplicate IDs."""

    prompts: list[Prompt] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                prompts.append(Prompt(id=payload["id"], text=payload["student_prompt"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid prompt at {path}:{line_number}") from error
    ids = [prompt.id for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate prompt ids in {path}")
    return tuple(prompts)


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


def build_chat_completion_payload(
    prompt: str, spec: GenerationSpec, *, sample_index: int
) -> dict[str, Any]:
    """Build one deterministic single-sample SGLang request."""

    if not 0 <= sample_index < spec.samples_per_prompt:
        raise ValueError("sample index is outside the generation specification")

    effective_prompt = render_generation_prompt(prompt, spec)
    payload = {
        "model": spec.model,
        "messages": [{"role": "user", "content": effective_prompt}],
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "min_p": spec.min_p,
        "max_tokens": spec.max_new_tokens,
        "n": 1,
        "seed": spec.seed + sample_index,
        "chat_template_kwargs": {"enable_thinking": spec.enable_thinking},
    }
    if spec.lora_path:
        payload["lora_path"] = spec.lora_path
    return payload


def render_generation_prompt(prompt: str, spec: GenerationSpec) -> str:
    """Render the exact user message seen by the generation model."""

    if not spec.response_instruction:
        return prompt
    return f"{prompt}\n\nResponse requirements:\n{spec.response_instruction}"


def generation_fingerprint(spec: GenerationSpec) -> str:
    """Return a stable hash of every generation control."""

    spec_payload = spec.model_dump(mode="json")
    if spec.lora_path is None:
        # Added after permanent teacher generation began; preserve fingerprints for
        # base-model shards while still fingerprinting every routed adapter.
        spec_payload.pop("lora_path")
    controls = {
        "sampling_strategy": SAMPLING_STRATEGY,
        "spec": spec_payload,
    }
    encoded = json.dumps(controls, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_artifact_identity(path: Path) -> str:
    """Hash every local file that can affect loaded inference behavior."""

    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"model artifact is not a directory: {root}")
    files = sorted(
        {
            candidate
            for pattern in INFERENCE_ARTIFACT_PATTERNS
            for candidate in root.glob(pattern)
            if candidate.is_file()
        },
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    has_config = any(
        candidate.name in {"config.json", "adapter_config.json"} for candidate in files
    )
    has_weights = any(
        candidate.name.endswith(".safetensors")
        or (candidate.name.startswith("pytorch_model") and candidate.name.endswith(".bin"))
        for candidate in files
    )
    if not has_config or not has_weights:
        raise ValueError(f"model artifact needs a config and weights: {root}")

    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(candidate.stat().st_size).encode())
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def parse_chat_completion_response(
    prompt_id: str,
    spec: GenerationSpec,
    response: Mapping[str, Any],
    *,
    sample_index: int,
) -> tuple[GenerationRecord, ...]:
    """Validate and normalize one single-sample SGLang response."""

    if not 0 <= sample_index < spec.samples_per_prompt:
        raise ValueError("sample index is outside the generation specification")

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(
            "expected one choice, received "
            f"{len(choices) if isinstance(choices, list) else 'an invalid value'}"
        )

    request_id = str(response.get("id", "")).strip()
    if not request_id:
        raise ValueError("SGLang response has no request id")
    response_model = str(response.get("model", spec.model))
    usage = response.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}

    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("SGLang response contains a non-object choice")
    if int(choice.get("index", -1)) != 0:
        raise ValueError("single-sample SGLang choice index must be zero")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("SGLang response choice has no text content")
    return (
        GenerationRecord(
            prompt_id=prompt_id,
            sample_index=sample_index,
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
        ),
    )


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


def ensure_generation_run_manifest(
    *,
    input_path: Path,
    output_dir: Path,
    prompts: Iterable[Prompt],
    spec: GenerationSpec,
    served_artifact_identity: str | None = None,
) -> Path:
    """Freeze the input prompts and generation controls for a resumable run."""

    prompt_tuple = tuple(prompts)
    manifest = {
        "schema_version": 1,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "config_hash": generation_fingerprint(spec),
        "generation": spec.model_dump(mode="json"),
        "prompts": [
            {
                "id": prompt.id,
                "text_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
            }
            for prompt in prompt_tuple
        ],
    }
    if served_artifact_identity is not None:
        if not served_artifact_identity.strip():
            raise ValueError("served artifact identity must be non-empty when provided")
        manifest["served_artifact_identity"] = served_artifact_identity
    destination = output_dir / "_metadata" / "run-manifest.json"
    if destination.exists():
        with destination.open(encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise ValueError(f"generation run changed for existing {destination}")
        return destination

    orphaned_shard = next(output_dir.glob("*.json"), None) if output_dir.exists() else None
    if orphaned_shard is not None:
        raise ValueError(
            f"generation shards exist without a run manifest: {orphaned_shard}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


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
    records = tuple(
        parse_chat_completion_response(
            prompt.id,
            spec,
            post_json(
                endpoint,
                build_chat_completion_payload(prompt.text, spec, sample_index=sample_index),
                timeout,
            ),
            sample_index=sample_index,
        )[0]
        for sample_index in range(spec.samples_per_prompt)
    )
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
