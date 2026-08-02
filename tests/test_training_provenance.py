import hashlib
import json

from novelty_distill.training.provenance import atomic_json, file_provenance


def test_file_provenance_binds_resolved_path_size_and_content(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"frozen":true}\n')

    assert file_provenance(artifact) == {
        "path": str(artifact.resolve()),
        "size_bytes": len(b'{"frozen":true}\n'),
        "sha256": hashlib.sha256(b'{"frozen":true}\n').hexdigest(),
    }


def test_atomic_json_writes_sorted_complete_payload(tmp_path) -> None:
    output = tmp_path / "metadata.json"

    atomic_json(output, {"z": 1, "a": {"value": True}})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "a": {"value": True},
        "z": 1,
    }
    assert output.read_text(encoding="utf-8").endswith("\n")
