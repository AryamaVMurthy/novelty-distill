from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_repository_sync_retries_transient_fetch_without_stale_fallback() -> None:
    script = (ROOT / "scripts" / "sync_repository.sh").read_text(encoding="utf-8")

    assert 'attempts="${REPOSITORY_FETCH_ATTEMPTS:-6}"' in script
    assert 'retry_seconds="${REPOSITORY_FETCH_RETRY_SECONDS:-10}"' in script
    assert "refusing stale-code fallback" in script
    assert 'merge --ff-only "origin/${repo_ref}"' in script


def test_every_network_synced_slurm_job_uses_bounded_helper() -> None:
    for path in (ROOT / "slurm").glob("*.sbatch"):
        script = path.read_text(encoding="utf-8")
        if "novelty-distill-sync.lock" not in script:
            continue
        assert "scripts/sync_repository.sh" in script, path.name
        assert ' fetch origin "${repo_ref}"' not in script, path.name
