import hashlib
import json

import pytest

from novelty_distill.evaluation.score_shards import validate_score_shard
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
                "records": [],
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
