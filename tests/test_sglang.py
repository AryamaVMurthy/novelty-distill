from pathlib import Path

from novelty_distill.generation.sglang import (
    GenerationRecord,
    GenerationSpec,
    Prompt,
    build_chat_completion_payload,
    generate_prompt,
    generation_fingerprint,
    load_prompt_shard,
    parse_chat_completion_response,
    pending_prompts,
    write_prompt_shard,
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

    payload = build_chat_completion_payload("Propose a hypothesis.", spec)

    assert payload == {
        "model": "Qwen/Qwen3-14B",
        "messages": [{"role": "user", "content": "Propose a hypothesis."}],
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 512,
        "n": 8,
        "seed": 17,
        "chat_template_kwargs": {"enable_thinking": False},
    }


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
            {"index": 1, "message": {"content": "second"}, "finish_reason": "stop"},
            {"index": 0, "message": {"content": "first"}, "finish_reason": "length"},
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }

    records = parse_chat_completion_response("tomato-7", spec, response)

    assert [record.sample_index for record in records] == [0, 1]
    assert [record.text for record in records] == ["first", "second"]
    assert records[0].finish_reason == "length"
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


def test_generate_prompt_calls_openai_endpoint_once_then_resumes(tmp_path: Path) -> None:
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
        return {
            "id": "chatcmpl-123",
            "model": spec.model,
            "choices": [
                {"index": 0, "message": {"content": "first"}, "finish_reason": "stop"},
                {"index": 1, "message": {"content": "second"}, "finish_reason": "stop"},
            ],
        }

    first = generate_prompt(
        prompt, spec, tmp_path, base_url="http://127.0.0.1:30000", timeout=120, post=post
    )
    second = generate_prompt(
        prompt, spec, tmp_path, base_url="http://127.0.0.1:30000", timeout=120, post=post
    )

    assert first == second
    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:30000/v1/chat/completions"
    assert calls[0][1] == build_chat_completion_payload(prompt.text, spec)
    assert calls[0][2] == 120
