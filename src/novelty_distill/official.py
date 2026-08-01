"""Pinned, external checkouts of official author repositories."""

import re
import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class OfficialRepository(BaseModel):
    """The immutable fields needed to fetch one official source."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str
    url: str = Field(pattern=r"^https://github\.com/")
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    subdirectory: str | None = None
    sparse_paths: tuple[str, ...] = ()


def load_official_repositories(path: Path) -> dict[str, OfficialRepository]:
    """Load all pinned Git repositories from the integration manifest."""

    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    repositories = manifest.get("repositories", {})
    return {
        name: OfficialRepository.model_validate({"name": name, **payload})
        for name, payload in repositories.items()
    }


def build_checkout_commands(
    repository: OfficialRepository, destination: Path
) -> tuple[tuple[str, ...], ...]:
    """Build the minimal Git commands for an immutable detached checkout."""

    checkout = destination / repository.name
    checkout_text = str(checkout)
    commands = [
        (
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository.url,
            checkout_text,
        ),
        ("git", "-C", checkout_text, "fetch", "--depth", "1", "origin", repository.commit),
    ]
    if repository.sparse_paths:
        commands.append(
            (
                "git",
                "-C",
                checkout_text,
                "sparse-checkout",
                "set",
                "--cone",
                *repository.sparse_paths,
            )
        )
    commands.append(("git", "-C", checkout_text, "checkout", "--detach", repository.commit))
    return tuple(commands)


def checkout_official_repository(repository: OfficialRepository, destination: Path) -> Path:
    """Create a pinned checkout, or verify an already-complete one without modifying it."""

    checkout = destination / repository.name
    if checkout.exists():
        head = subprocess.run(
            ("git", "-C", str(checkout), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(checkout), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if not re.fullmatch(r"[0-9a-f]{40}", head) or head != repository.commit:
            raise RuntimeError(
                f"existing checkout {checkout} is at {head}, not {repository.commit}"
            )
        if status:
            raise RuntimeError(f"existing checkout {checkout} has local changes")
        return checkout

    destination.mkdir(parents=True, exist_ok=True)
    for command in build_checkout_commands(repository, destination):
        subprocess.run(command, check=True)
    return checkout
