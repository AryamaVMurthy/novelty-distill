import json
from pathlib import Path
from typing import Any

from novelty_distill.tracking.wandb_export import export_wandb_records


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str | None]] = []

    def add_file(self, path: str, name: str | None = None) -> None:
        self.files.append((path, name))


class FakeRun:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.summary: dict[str, Any] = {}
        self.logged: list[dict[str, Any]] = []
        self.artifacts: list[FakeArtifact] = []
        self.finished = False

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(payload)

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.runs: list[FakeRun] = []

    @staticmethod
    def Table(*, columns: list[str], data: list[list[Any]]) -> dict[str, Any]:
        return {"columns": columns, "data": data}

    def init(self, **kwargs: Any) -> FakeRun:
        run = FakeRun(kwargs)
        self.runs.append(run)
        return run


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_offline_export_logs_configs_metrics_tables_and_metadata_artifacts(
    tmp_path: Path,
) -> None:
    training = tmp_path / "run_metadata.json"
    evaluation = tmp_path / "evaluation.json"
    manifest = tmp_path / "wandb-export.json"
    _write(
        training,
        {
            "baseline_id": "B1",
            "backend": "trl_sft",
            "model": "Qwen/Qwen3-4B",
            "revision": "a" * 40,
            "seed": 17,
            "git_commit": "b" * 40,
            "metrics": {"train_loss": 1.25, "train_runtime": 12.0},
            "example_ids": ["must-not-enter-config"],
        },
    )
    _write(
        evaluation,
        {
            "schema_version": 1,
            "git_commit": "c" * 40,
            "num_prompts": 2,
            "viable_semantic_yield": {"status": "secondary_descriptive"},
            "overall": {"student_quality_mean": 0.75, "teacher_mode_recall": 0.5},
            "prompt_metrics": {"large": {"student_quality_mean": 0.75}},
        },
    )
    fake = FakeWandb()

    result = export_wandb_records(
        (training, evaluation),
        manifest_path=manifest,
        wandb_module=fake,
        project="novelty-distill",
        entity=None,
        mode="offline",
        run_dir=tmp_path / "runs",
    )

    assert result["exported"] == 2
    assert result["skipped"] == 0
    assert len(fake.runs) == 2
    training_run, evaluation_run = fake.runs
    assert training_run.kwargs["mode"] == "offline"
    assert training_run.kwargs["reinit"] == "finish_previous"
    assert training_run.kwargs["config"]["baseline_id"] == "B1"
    assert "example_ids" not in training_run.kwargs["config"]
    assert training_run.summary["train_loss"] == 1.25
    assert training_run.logged[0]["metrics"]["columns"] == ["metric", "value"]
    assert training_run.artifacts[0].files == [(str(training.resolve()), training.name)]
    assert evaluation_run.summary["student_quality_mean"] == 0.75
    assert evaluation_run.kwargs["config"]["viable_semantic_yield"] == {
        "status": "secondary_descriptive"
    }
    assert all(run.finished for run in fake.runs)
    assert json.loads(manifest.read_text())["schema_version"] == 1


def test_export_manifest_makes_identical_backfill_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "run_metadata.json"
    manifest = tmp_path / "wandb-export.json"
    _write(source, {"baseline_id": "B3", "metrics": {"train_loss": 0.5}})
    first = FakeWandb()
    export_wandb_records(
        (source,),
        manifest_path=manifest,
        wandb_module=first,
        project="novelty-distill",
        entity=None,
        mode="offline",
        run_dir=tmp_path / "runs",
    )
    second = FakeWandb()

    result = export_wandb_records(
        (source,),
        manifest_path=manifest,
        wandb_module=second,
        project="novelty-distill",
        entity=None,
        mode="offline",
        run_dir=tmp_path / "runs",
    )

    assert result["exported"] == 0
    assert result["skipped"] == 1
    assert second.runs == []


def test_final_analysis_jobs_export_offline_tracking_snapshots() -> None:
    primary = Path("slurm/analyze_evaluation_matrix.sbatch").read_text(encoding="utf-8")
    exploratory = Path("slurm/analyze_exploratory_drkl.sbatch").read_text(
        encoding="utf-8"
    )
    data_requirements = Path("environments/data.in").read_text(encoding="utf-8")
    exporter = Path("scripts/export_wandb_snapshot.py").read_text(encoding="utf-8")

    assert "wandb==0.22.3" in data_requirements
    for script in (primary, exploratory):
        assert "scripts/export_wandb_snapshot.py" in script
        assert "wandb-export-manifest.json" in script
        assert '--run-dir "${scratch_root}/wandb"' in script
    assert "C3-tomato1k-seed17-l896-4gpu/run_metadata.json" in primary
    for variable in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        assert f'os.environ.setdefault("{variable}"' in exporter
