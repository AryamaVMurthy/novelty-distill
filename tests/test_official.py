from pathlib import Path

import yaml

from novelty_distill.official import build_checkout_commands, load_official_repositories


def test_official_repositories_are_pinned_and_checkout_is_detached() -> None:
    repositories = load_official_repositories(Path("third_party/manifest.yaml"))

    assert {"trl", "sglang", "opsd", "gem", "minillm", "distillm"} <= repositories.keys()
    assert all(
        repository.url.startswith("https://github.com/") for repository in repositories.values()
    )
    assert all(len(repository.commit) == 40 for repository in repositories.values())

    commands = build_checkout_commands(repositories["opsd"], Path("/scratch/project/official"))

    assert commands[-1] == (
        "git",
        "-C",
        "/scratch/project/official/opsd",
        "checkout",
        "--detach",
        "7448751f307a9cdbcc1246dd1565a1a605b443df",
    )


def test_large_official_repository_uses_manifest_sparse_paths() -> None:
    repositories = load_official_repositories(Path("third_party/manifest.yaml"))

    commands = build_checkout_commands(
        repositories["inspect_evals"], Path("/scratch/project/official")
    )

    assert commands[-2] == (
        "git",
        "-C",
        "/scratch/project/official/inspect_evals",
        "sparse-checkout",
        "set",
        "--no-cone",
        "packages/novelty_bench",
        "src",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "NOTICE",
    )
    assert commands[-1][-3:] == (
        "checkout",
        "--detach",
        "6a35510e530f236fd1dbcd9df888f01937c8494a",
    )


def test_installed_official_benchmarks_have_verified_license_provenance() -> None:
    manifest = yaml.safe_load(Path("third_party/manifest.yaml").read_text(encoding="utf-8"))
    repositories = manifest["repositories"]

    assert repositories["inspect_evals"]["license"] == "MIT"
    assert repositories["inspect_evals"]["license_evidence"] == "LICENSE and pyproject.toml"
    assert repositories["hypospace"]["license"] == "MIT"
    assert repositories["hypospace"]["license_evidence"] == "README.md license section"
