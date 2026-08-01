from pathlib import Path

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
