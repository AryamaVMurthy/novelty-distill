import hashlib

from novelty_distill.training.provenance import file_provenance


def test_file_provenance_binds_resolved_path_size_and_content(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"frozen":true}\n')

    assert file_provenance(artifact) == {
        "path": str(artifact.resolve()),
        "size_bytes": len(b'{"frozen":true}\n'),
        "sha256": hashlib.sha256(b'{"frozen":true}\n').hexdigest(),
    }
