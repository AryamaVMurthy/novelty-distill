#!/usr/bin/env python3
"""Analyze agreement, repeat reliability, and method means for independent judgments."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from novelty_distill.evaluation.independent_judge import (
    CalibrationEntry,
    ExternalJudgeScore,
    analyze_independent_judgments,
    independent_judge_protocol_hash,
    parse_independent_judge_response,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet_bytes = args.packet.read_bytes()
    packet = json.loads(packet_bytes)
    protocol_hash = independent_judge_protocol_hash()
    if packet.get("protocol_hash") != protocol_hash:
        raise ValueError("calibration packet protocol hash does not match this checkout")
    entries = [CalibrationEntry.model_validate(item) for item in packet["entries"]]
    ratings: dict[str, ExternalJudgeScore] = {}
    response_hashes: dict[str, str] = {}
    provider_models: set[str] = set()
    request_ids: set[str] = set()
    finish_reasons: Counter[str] = Counter()
    output_wrappers: Counter[str] = Counter()
    provider_usage = {
        "responses": len(entries),
        "responses_with_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "responses_with_estimated_cost": 0,
    }
    estimated_costs: list[float] = []
    for entry in entries:
        path = args.responses / f"{entry.blind_id}.json"
        raw = path.read_bytes()
        payload = json.loads(raw)
        if (
            payload.get("blind_id") != entry.blind_id
            or payload.get("text_sha256") != entry.text_sha256
            or payload.get("protocol_hash") != protocol_hash
        ):
            raise ValueError(f"response identity mismatch at {path}")
        saved_score = ExternalJudgeScore.model_validate(payload["parsed_score"])
        parsed = parse_independent_judge_response(payload["raw_response"])
        if parsed.score != saved_score or parsed.model != payload.get("provider_model"):
            raise ValueError(f"saved response fields do not match raw provider response at {path}")
        if parsed.request_id in request_ids:
            raise ValueError(f"duplicate provider request id at {path}")
        request_ids.add(parsed.request_id)
        ratings[entry.blind_id] = saved_score
        response_hashes[entry.blind_id] = hashlib.sha256(raw).hexdigest()
        provider_models.add(str(payload["provider_model"]))
        choices = payload["raw_response"].get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], Mapping)
        ):
            raise ValueError(f"provider choices are invalid at {path}")
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is None:
            finish_reason = "missing"
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ValueError(f"provider finish reason is invalid at {path}")
        finish_reasons[finish_reason] += 1
        message = choices[0].get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise ValueError(f"provider message is invalid at {path}")
        wrapper = (
            "single_json_fence" if message["content"].strip().startswith("```") else "raw_json"
        )
        output_wrappers[wrapper] += 1
        usage = payload["raw_response"].get("usage", {})
        if not isinstance(usage, Mapping):
            raise ValueError(f"provider usage is not an object at {path}")
        if usage:
            provider_usage["responses_with_usage"] += 1
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"provider usage {field!r} is invalid at {path}")
            provider_usage[field] += value
        estimated_cost = usage.get("estimated_cost")
        if estimated_cost is not None:
            if (
                isinstance(estimated_cost, bool)
                or not isinstance(estimated_cost, int | float)
                or not math.isfinite(estimated_cost)
                or estimated_cost < 0
            ):
                raise ValueError(f"provider estimated cost is invalid at {path}")
            provider_usage["responses_with_estimated_cost"] += 1
            estimated_costs.append(float(estimated_cost))

    provider_usage["estimated_cost_usd"] = math.fsum(estimated_costs)
    provider_transport = {
        "unique_request_ids": len(request_ids),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "length_stop_rate": finish_reasons["length"] / len(entries),
        "output_wrapper_counts": dict(sorted(output_wrappers.items())),
    }

    summary = analyze_independent_judgments(entries=entries, ratings=ratings)
    summary["provenance"] = {
        "packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "protocol_hash": protocol_hash,
        "provider_models": sorted(provider_models),
        "provider_transport": provider_transport,
        "provider_usage": provider_usage,
        "response_hashes": response_hashes,
    }
    _atomic_json(args.output, summary)
    print(json.dumps({"output": str(args.output), "counts": summary["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
