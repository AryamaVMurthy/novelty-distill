#!/usr/bin/env python3
"""Run a resumable blinded judge calibration through DeepInfra's OpenAI API."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.independent_judge import (
    CalibrationEntry,
    build_independent_judge_payload,
    independent_judge_protocol_hash,
    parse_independent_judge_response,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _post(*, endpoint: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read(1000).decode(errors="replace")
        raise RuntimeError(f"DeepInfra HTTP {error.code}: {detail}") from error
    if not isinstance(body, dict):
        raise ValueError("DeepInfra returned a non-object response")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    parser.add_argument(
        "--endpoint", default="https://api.deepinfra.com/v1/openai/chat/completions"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0 or args.attempts <= 0:
        raise ValueError("concurrency and attempts must be positive")

    packet_bytes = args.packet.read_bytes()
    packet = json.loads(packet_bytes)
    protocol_hash = independent_judge_protocol_hash()
    if packet.get("protocol_hash") != protocol_hash:
        raise ValueError("calibration packet protocol hash does not match this checkout")
    entries = [CalibrationEntry.model_validate(item) for item in packet["entries"]]
    if len({entry.blind_id for entry in entries}) != len(entries):
        raise ValueError("calibration packet has duplicate blind ids")
    if args.dry_run:
        first_payload = build_independent_judge_payload(
            entries[0], model=args.model, max_tokens=args.max_tokens
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "entries": len(entries),
                    "model": args.model,
                    "first_payload_sha256": hashlib.sha256(
                        json.dumps(first_payload, sort_keys=True).encode()
                    ).hexdigest(),
                },
                sort_keys=True,
            )
        )
        return

    api_key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY is required and must not be passed on the command line"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def judge(entry: CalibrationEntry) -> tuple[str, str]:
        output_path = args.output_dir / f"{entry.blind_id}.json"
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                existing.get("blind_id") != entry.blind_id
                or existing.get("text_sha256") != entry.text_sha256
                or existing.get("protocol_hash") != protocol_hash
                or existing.get("request_model") != args.model
            ):
                raise ValueError(f"existing output identity mismatch at {output_path}")
            parse_independent_judge_response(existing["raw_response"])
            return entry.blind_id, "reused"
        payload = build_independent_judge_payload(
            entry, model=args.model, max_tokens=args.max_tokens
        )
        last_error: Exception | None = None
        for attempt in range(1, args.attempts + 1):
            try:
                raw_response = _post(
                    endpoint=args.endpoint,
                    api_key=api_key,
                    payload=payload,
                    timeout=args.timeout,
                )
                parsed = parse_independent_judge_response(raw_response)
                _atomic_json(
                    output_path,
                    {
                        "schema_version": 1,
                        "blind_id": entry.blind_id,
                        "text_sha256": entry.text_sha256,
                        "protocol_hash": protocol_hash,
                        "request_model": args.model,
                        "provider_model": parsed.model,
                        "parsed_score": parsed.score.model_dump(mode="json"),
                        "raw_response": raw_response,
                    },
                )
                return entry.blind_id, "created"
            except (OSError, RuntimeError, ValueError) as error:
                last_error = error
                if attempt < args.attempts:
                    time.sleep(min(8, 2 ** (attempt - 1)))
        raise RuntimeError(f"judge failed for {entry.blind_id}: {last_error}")

    counts = {"created": 0, "reused": 0}
    completed = 0
    entry_iterator = iter(entries)
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    in_flight: dict[Future[tuple[str, str]], str] = {}

    def submit_next() -> bool:
        try:
            entry = next(entry_iterator)
        except StopIteration:
            return False
        in_flight[executor.submit(judge, entry)] = entry.blind_id
        return True

    try:
        for _ in range(min(args.concurrency, len(entries))):
            submit_next()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            batch = [future.result() for future in done]
            for future in done:
                del in_flight[future]
            for blind_id, status in batch:
                completed += 1
                counts[status] += 1
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(entries),
                            "blind_id": blind_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            for _ in batch:
                submit_next()
    except BaseException:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    manifest = {
        "schema_version": 1,
        "packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "protocol_hash": protocol_hash,
        "request_model": args.model,
        "endpoint": args.endpoint,
        "entries": len(entries),
        "counts": counts,
    }
    _atomic_json(args.output_dir / "_run_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
