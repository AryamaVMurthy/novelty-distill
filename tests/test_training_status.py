import json

import pytest

from novelty_distill.training.status import adapter_run_complete


def test_adapter_run_completion_is_strict_and_fail_closed(tmp_path) -> None:
    metadata = tmp_path / "run_metadata.json"
    assert not adapter_run_complete(metadata, baseline_id="D1", max_steps=125)

    final = tmp_path / "final"
    final.mkdir()
    (final / "adapter_config.json").write_text("{}", encoding="utf-8")
    (final / "adapter_model.safetensors").write_bytes(b"weights")
    metadata.write_text(
        json.dumps({"baseline_id": "D1", "max_steps": 125, "final_dir": str(final)}),
        encoding="utf-8",
    )
    assert adapter_run_complete(metadata, baseline_id="D1", max_steps=125)

    with pytest.raises(ValueError, match="stale or incompatible"):
        adapter_run_complete(metadata, baseline_id="D2", max_steps=125)


def test_adapter_run_completion_rejects_external_or_missing_final_files(tmp_path) -> None:
    metadata = tmp_path / "run_metadata.json"
    metadata.write_text(
        json.dumps(
            {"baseline_id": "D1", "max_steps": 125, "final_dir": str(tmp_path.parent)}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        adapter_run_complete(metadata, baseline_id="D1", max_steps=125)
