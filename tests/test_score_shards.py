import hashlib
import json

import pytest

from novelty_distill.evaluation.score_shards import load_score_shard, validate_score_shard
from novelty_distill.evaluation.teacher_annotation import JudgeSpec


def test_validate_score_shard_distinguishes_missing_current_and_stale(tmp_path) -> None:
    judge = JudgeSpec(model="judge", revision="a" * 40)
    score_path = tmp_path / "score.json"
    hashes = [hashlib.sha256(b"response").hexdigest()]

    assert not validate_score_shard(
        score_path, prompt_id="p1", text_hashes=hashes, judge=judge
    )
    score_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "prompt_id": "p1",
                "text_hashes": hashes,
                "judge": judge.model_dump(mode="json"),
                "records": [
                    {
                        "prompt_id": "p1",
                        "sample_index": 0,
                        "text": "response",
                        "finish_reason": "stop",
                        "completion_tokens": 4,
                        "dimensions": {
                            "relevance": 5,
                            "feasibility": 4,
                            "soundness": 3,
                            "clarity": 2,
                            "instruction_compliance": 1,
                        },
                        "quality_score": 0.5,
                        "request_id": "request-1",
                        "model": "judge",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert validate_score_shard(
        score_path, prompt_id="p1", text_hashes=hashes, judge=judge
    )

    score_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="stale or incompatible"):
        validate_score_shard(
            score_path, prompt_id="p1", text_hashes=hashes, judge=judge
        )


def test_load_score_shard_rejects_quality_inconsistent_with_dimensions(tmp_path) -> None:
    path = tmp_path / "score.json"
    text = "response"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "prompt_id": "p1",
                "text_hashes": [hashlib.sha256(text.encode()).hexdigest()],
                "judge": {"model": "judge", "revision": "a" * 40, "max_tokens": 128},
                "records": [
                    {
                        "prompt_id": "p1",
                        "sample_index": 0,
                        "text": text,
                        "finish_reason": "stop",
                        "completion_tokens": 4,
                        "dimensions": {
                            "relevance": 5,
                            "feasibility": 5,
                            "soundness": 5,
                            "clarity": 5,
                            "instruction_compliance": 5,
                        },
                        "quality_score": 0.0,
                        "request_id": "request-1",
                        "model": "judge",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quality score"):
        load_score_shard(path, samples_per_prompt=1)
