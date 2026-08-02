"""Repository provenance shared by generated research artifacts."""

import re
import subprocess
from pathlib import Path


def repository_commit(repo_root: Path) -> str:
    """Resolve and validate the exact Git commit containing the executing script."""

    result = subprocess.run(
        ("git", "-C", str(repo_root.resolve()), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"invalid repository commit {commit!r}")
    return commit
