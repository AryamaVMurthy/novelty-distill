import hashlib
import json
from pathlib import Path

import pytest

from novelty_distill.data.semantic_seeds import (
    GaussianSeedSpec,
    condition_prompt,
    gaussian_seed_values,
)
from novelty_distill.generation.sglang import (
    GenerationRecord,
    GenerationSpec,
    Prompt,
    bootstrap_generation_shards,
    build_chat_completion_payload,
    ensure_generation_run_manifest,
    generate_prompt,
    generation_fingerprint,
    load_prompt_shard,
    model_artifact_identity,
    parse_chat_completion_response,
    pending_prompts,
    shard_prompts,
    write_prompt_shard,
)


def test_prompt_shards_are_disjoint_complete_and_deterministic() -> None:
    prompts = tuple(Prompt(id=f"p{index}", text=f"prompt {index}") for index in range(11))

    shards = tuple(shard_prompts(prompts, num_shards=4, shard_index=index) for index in range(4))

    assert [len(shard) for shard in shards] == [3, 3, 3, 2]
    assert tuple(prompt for shard in shards for prompt in shard) != prompts
    assert {prompt.id for shard in shards for prompt in shard} == {prompt.id for prompt in prompts}
    assert sum(len(shard) for shard in shards) == len(prompts)
    assert shard_prompts(prompts, num_shards=4, shard_index=2) == shards[2]


@pytest.mark.parametrize(
    ("num_shards", "shard_index"),
    ((0, 0), (2, -1), (2, 2)),
)
def test_prompt_sharding_rejects_invalid_coordinates(num_shards: int, shard_index: int) -> None:
    with pytest.raises(ValueError, match="shard"):
        shard_prompts(
            (Prompt(id="p0", text="prompt"),),
            num_shards=num_shards,
            shard_index=shard_index,
        )


def test_qwen_payload_disables_thinking_and_fixes_sampling_controls() -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        samples_per_prompt=8,
        seed=17,
    )

    payload = build_chat_completion_payload("Propose a hypothesis.", spec, sample_index=3)

    assert payload == {
        "model": "Qwen/Qwen3-14B",
        "messages": [{"role": "user", "content": "Propose a hypothesis."}],
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_generation_response_instruction_is_part_of_the_effective_prompt_and_fingerprint() -> None:
    base = GenerationSpec(
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=512,
        samples_per_prompt=8,
        seed=17,
    )
    concise = base.model_copy(update={"response_instruction": "Return at most 300 words."})

    payload = build_chat_completion_payload("Propose a hypothesis.", concise, sample_index=0)

    assert payload["messages"] == [
        {
            "role": "user",
            "content": "Propose a hypothesis.\n\nResponse requirements:\nReturn at most 300 words.",
        }
    ]
    assert generation_fingerprint(concise) != generation_fingerprint(base)


def test_gaussian_seed_is_rendered_before_response_requirements_and_fingerprinted() -> None:
    base = GenerationSpec(
        model="Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        temperature=0.2,
        top_p=0.8,
        max_new_tokens=512,
        samples_per_prompt=4,
        seed=17,
        response_instruction="Return one concise hypothesis.",
    )
    seed_spec = GaussianSeedSpec(dimensions=4, bins=5, scale=1.0, salt="v1")
    seeded = base.model_copy(update={"input_seed": seed_spec})
    values = gaussian_seed_values(
        prompt_id="paper-seed", sample_index=2, generation_seed=17, spec=seed_spec
    )

    payload = build_chat_completion_payload(
        "Propose a hypothesis.", seeded, sample_index=2, prompt_id="paper-seed"
    )

    assert payload["messages"][0]["content"] == (
        f"{condition_prompt('Propose a hypothesis.', values=values)}\n\n"
        "Response requirements:\nReturn one concise hypothesis."
    )
    assert payload["seed"] == 19
    assert generation_fingerprint(seeded) != generation_fingerprint(base)


def test_seeded_payload_requires_stable_prompt_id() -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        temperature=0.2,
        top_p=0.8,
        max_new_tokens=512,
        samples_per_prompt=4,
        seed=17,
        input_seed=GaussianSeedSpec(dimensions=4, bins=5, scale=1.0, salt="v1"),
    )

    with pytest.raises(ValueError, match="prompt ID"):
        build_chat_completion_payload("Propose a hypothesis.", spec, sample_index=0)


def test_lora_adapter_is_routed_explicitly_and_fingerprinted() -> None:
    base = GenerationSpec(
        model="Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=512,
        samples_per_prompt=16,
        seed=17,
    )
    adapted = base.model_copy(update={"lora_path": "D1-seed17"})

    payload = build_chat_completion_payload("Propose a hypothesis.", adapted, sample_index=0)

    assert payload["model"] == base.model
    assert payload["lora_path"] == "D1-seed17"
    assert "lora_path" not in build_chat_completion_payload(
        "Propose a hypothesis.", base, sample_index=0
    )
    assert generation_fingerprint(adapted) != generation_fingerprint(base)


def test_absent_lora_field_preserves_pre_lora_generation_fingerprint() -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=512,
        samples_per_prompt=8,
        seed=17,
        response_instruction="Return one concise hypothesis.",
    )
    legacy_spec = spec.model_dump(mode="json")
    legacy_spec.pop("lora_path")
    legacy_spec.pop("input_seed")
    encoded = json.dumps(
        {"sampling_strategy": "single-request-per-sample-v1", "spec": legacy_spec},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert generation_fingerprint(spec) == hashlib.sha256(encoded).hexdigest()


def test_response_is_parsed_into_auditable_sample_records() -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        samples_per_prompt=2,
        seed=17,
    )
    response = {
        "id": "chatcmpl-123",
        "model": "Qwen/Qwen3-14B",
        "choices": [
            {"index": 0, "message": {"content": "second"}, "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }

    records = parse_chat_completion_response("tomato-7", spec, response, sample_index=1)

    assert [record.sample_index for record in records] == [1]
    assert [record.text for record in records] == ["second"]
    assert records[0].finish_reason == "stop"
    assert records[0].request_id == "chatcmpl-123"
    assert records[0].config_hash == generation_fingerprint(spec)
    assert records[0].prompt_tokens == 11
    assert records[0].completion_tokens == 7


def test_atomic_prompt_shards_make_generation_resumable(tmp_path: Path) -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        samples_per_prompt=2,
        seed=17,
    )
    records = tuple(
        GenerationRecord(
            prompt_id="unsafe/id",
            sample_index=index,
            text=f"sample-{index}",
            finish_reason="stop",
            model=spec.model,
            request_id="chatcmpl-123",
            config_hash=generation_fingerprint(spec),
        )
        for index in range(2)
    )

    shard = write_prompt_shard(tmp_path, records, spec)

    assert shard.parent == tmp_path
    assert "/" not in shard.name
    assert load_prompt_shard(shard, spec) == records
    prompts = (Prompt(id="unsafe/id", text="done"), Prompt(id="tomato-8", text="pending"))
    assert pending_prompts(prompts, tmp_path, spec) == (prompts[1],)


def test_generation_manifest_rejects_changed_prompt_text(tmp_path: Path) -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        samples_per_prompt=2,
        seed=17,
    )
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text('{"id":"p1","student_prompt":"first"}\n', encoding="utf-8")
    output_dir = tmp_path / "generation"

    manifest = ensure_generation_run_manifest(
        input_path=input_path,
        output_dir=output_dir,
        prompts=(Prompt(id="p1", text="first"),),
        spec=spec,
    )

    assert manifest == output_dir / "_metadata" / "run-manifest.json"
    input_path.write_text('{"id":"p1","student_prompt":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        ensure_generation_run_manifest(
            input_path=input_path,
            output_dir=output_dir,
            prompts=(Prompt(id="p1", text="changed"),),
            spec=spec,
        )


def test_larger_generation_run_bootstraps_exact_subset_without_regeneration(
    tmp_path: Path,
) -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        samples_per_prompt=2,
        seed=17,
    )
    source_input = tmp_path / "source.jsonl"
    target_input = tmp_path / "target.jsonl"
    source_input.write_text('{"id":"p1","student_prompt":"first"}\n', encoding="utf-8")
    target_input.write_text(
        '{"id":"p1","student_prompt":"first"}\n{"id":"p2","student_prompt":"second"}\n',
        encoding="utf-8",
    )
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    prompts = (Prompt(id="p1", text="first"),)
    ensure_generation_run_manifest(
        input_path=source_input, output_dir=source_dir, prompts=prompts, spec=spec
    )
    ensure_generation_run_manifest(
        input_path=target_input,
        output_dir=target_dir,
        prompts=(*prompts, Prompt(id="p2", text="second")),
        spec=spec,
    )
    records = tuple(
        GenerationRecord(
            prompt_id="p1",
            sample_index=index,
            text=f"sample-{index}",
            finish_reason="stop",
            model=spec.model,
            request_id=f"request-{index}",
            config_hash=generation_fingerprint(spec),
        )
        for index in range(2)
    )
    write_prompt_shard(source_dir, records, spec)

    assert bootstrap_generation_shards(source_dir=source_dir, target_dir=target_dir) == (1, 0)
    assert bootstrap_generation_shards(source_dir=source_dir, target_dir=target_dir) == (0, 1)
    assert load_prompt_shard(next(target_dir.glob("*.json")), spec) == records


def test_generation_manifest_binds_local_model_artifact_bytes(tmp_path: Path) -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=512,
        samples_per_prompt=16,
        seed=17,
    )
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text('{"rank":16}\n', encoding="utf-8")
    weights = artifact / "adapter_model.safetensors"
    weights.write_bytes(b"first-weights")
    (artifact / "run_metadata.json").write_text('{"note":"ignored"}\n', encoding="utf-8")
    first_identity = model_artifact_identity(artifact)
    (artifact / "run_metadata.json").write_text('{"note":"changed"}\n', encoding="utf-8")
    assert model_artifact_identity(artifact) == first_identity

    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text('{"id":"p1","student_prompt":"first"}\n', encoding="utf-8")
    output_dir = tmp_path / "generation"
    ensure_generation_run_manifest(
        input_path=input_path,
        output_dir=output_dir,
        prompts=(Prompt(id="p1", text="first"),),
        spec=spec,
        served_artifact_identity=first_identity,
    )
    payload = json.loads(
        (output_dir / "_metadata" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert payload["served_artifact_identity"] == first_identity

    weights.write_bytes(b"second-weights")
    second_identity = model_artifact_identity(artifact)
    assert second_identity != first_identity
    with pytest.raises(ValueError, match="changed"):
        ensure_generation_run_manifest(
            input_path=input_path,
            output_dir=output_dir,
            prompts=(Prompt(id="p1", text="first"),),
            spec=spec,
            served_artifact_identity=second_identity,
        )


def test_generation_manifest_rejects_orphaned_existing_shards(tmp_path: Path) -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=512,
        samples_per_prompt=16,
        seed=17,
    )
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text('{"id":"p1","student_prompt":"first"}\n', encoding="utf-8")
    output_dir = tmp_path / "generation"
    output_dir.mkdir()
    (output_dir / "orphan.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without a run manifest"):
        ensure_generation_run_manifest(
            input_path=input_path,
            output_dir=output_dir,
            prompts=(Prompt(id="p1", text="first"),),
            spec=spec,
        )


def test_generate_prompt_calls_one_seeded_request_per_sample_then_resumes(tmp_path: Path) -> None:
    spec = GenerationSpec(
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        samples_per_prompt=2,
        seed=17,
    )
    prompt = Prompt(id="tomato-7", text="Propose a hypothesis.")
    calls: list[tuple[str, dict[str, object], float]] = []

    def post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        calls.append((url, payload, timeout))
        seed = int(payload["seed"])
        return {
            "id": f"chatcmpl-{seed}",
            "model": spec.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"content": f"sample-{seed}"},
                    "finish_reason": "stop",
                },
            ],
        }

    first = generate_prompt(
        prompt, spec, tmp_path, base_url="http://127.0.0.1:30000", timeout=120, post=post
    )
    second = generate_prompt(
        prompt, spec, tmp_path, base_url="http://127.0.0.1:30000", timeout=120, post=post
    )

    assert first == second
    assert [record.text for record in first] == ["sample-17", "sample-18"]
    assert len(calls) == 2
    assert calls[0][0] == "http://127.0.0.1:30000/v1/chat/completions"
    assert calls[0][1] == build_chat_completion_payload(prompt.text, spec, sample_index=0)
    assert calls[1][1] == build_chat_completion_payload(prompt.text, spec, sample_index=1)
    assert calls[0][2] == 120
