from pathlib import Path

from novelty_distill.provenance import repository_commit


def test_repository_commit_resolves_current_pinned_source() -> None:
    root = Path(__file__).resolve().parents[1]

    commit = repository_commit(root)

    assert len(commit) == 40
    assert set(commit) <= set("0123456789abcdef")
