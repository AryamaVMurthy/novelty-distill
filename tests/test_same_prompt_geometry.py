import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from novelty_distill.evaluation.embeddings import EmbeddingCache
from novelty_distill.evaluation.same_prompt_geometry import (
    analyze_same_prompt_geometry,
    summarize_same_prompt_geometry_metrics,
)


def _prompt_vectors() -> dict[str, dict[str, tuple[tuple[float, ...], ...]]]:
    diagonal = 2**-0.5
    per_prompt = {
        "A1": ((1.0, 0.0),),
        "A3": ((0.0, 1.0),),
        "A0": ((diagonal, diagonal), (1.0, 0.0)),
        "B1": ((0.0, 1.0), (0.0, 1.0)),
    }
    return {method: {"p1": values, "p2": values} for method, values in per_prompt.items()}


def test_same_prompt_geometry_separates_teacher_human_affinity_and_concentration() -> None:
    result = analyze_same_prompt_geometry(
        _prompt_vectors(),
        human_method="A3",
        teacher_method="A1",
        base_method="A0",
        resamples=100,
        seed=17,
    )

    assert result["num_prompts"] == 2
    assert result["reference"]["teacher_human_cosine_mean"] == pytest.approx(0.0)
    a0 = result["methods"]["A0"]["metrics"]
    assert a0["teacher_cosine_mean"]["estimate"] == pytest.approx(
        (1 + 2**-0.5) / 2
    )
    assert a0["human_cosine_mean"]["estimate"] == pytest.approx(2**-0.5 / 2)
    assert a0["teacher_minus_human_affinity"]["estimate"] == pytest.approx(0.5)
    assert a0["within_method_pair_cosine"]["estimate"] == pytest.approx(2**-0.5)

    b1 = result["methods"]["B1"]["metrics"]
    assert b1["teacher_cosine_mean"]["estimate"] == pytest.approx(0.0)
    assert b1["human_cosine_mean"]["estimate"] == pytest.approx(1.0)
    assert b1["teacher_minus_human_affinity"]["estimate"] == pytest.approx(-1.0)
    assert b1["within_method_pair_cosine"]["estimate"] == pytest.approx(1.0)
    assert b1["human_cosine_delta_vs_base"]["estimate"] == pytest.approx(
        1 - 2**-0.5 / 2
    )
    assert all(
        math.isclose(metric["estimate"], metric["ci_low"], abs_tol=1e-12)
        and math.isclose(metric["estimate"], metric["ci_high"], abs_tol=1e-12)
        for metric in b1.values()
    )


def test_same_prompt_geometry_requires_aligned_prompts_and_multi_sample_candidates() -> None:
    vectors = _prompt_vectors()
    vectors["B1"] = {"p1": ((0.0, 1.0), (0.0, 1.0))}
    with pytest.raises(ValueError, match="same prompt population"):
        analyze_same_prompt_geometry(
            vectors,
            human_method="A3",
            teacher_method="A1",
            base_method="A0",
            resamples=10,
            seed=17,
        )

    vectors = _prompt_vectors()
    vectors["B1"] = {"p1": ((0.0, 1.0),), "p2": ((0.0, 1.0),)}
    with pytest.raises(ValueError, match="at least two samples"):
        analyze_same_prompt_geometry(
            vectors,
            human_method="A3",
            teacher_method="A1",
            base_method="A0",
            resamples=10,
            seed=17,
        )


def test_same_prompt_geometry_rejects_nonfinite_or_dimension_mismatched_vectors() -> None:
    vectors = _prompt_vectors()
    vectors["B1"]["p1"] = ((float("nan"), 1.0), (0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        analyze_same_prompt_geometry(
            vectors,
            human_method="A3",
            teacher_method="A1",
            base_method="A0",
            resamples=10,
            seed=17,
        )

    vectors = _prompt_vectors()
    vectors["B1"]["p1"] = ((0.0, 1.0, 2.0), (0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="dimension"):
        analyze_same_prompt_geometry(
            vectors,
            human_method="A3",
            teacher_method="A1",
            base_method="A0",
            resamples=10,
            seed=17,
        )


def test_same_prompt_geometry_is_secondary_cache_only_and_wired_into_final_analysis() -> None:
    config = yaml.safe_load(
        Path("configs/evaluation/same_prompt_geometry.yaml").read_text(encoding="utf-8")
    )
    script = Path("scripts/analyze_same_prompt_geometry.py").read_text(encoding="utf-8")
    slurm = Path("slurm/analyze_evaluation_matrix.sbatch").read_text(encoding="utf-8")
    exposure_slurm = Path("slurm/analyze_exposure_sensitivity.sbatch").read_text(
        encoding="utf-8"
    )
    promoted_slurm = Path("slurm/analyze_promoted_matrix.sbatch").read_text(
        encoding="utf-8"
    )
    drkl_slurm = Path("slurm/analyze_exploratory_drkl.sbatch").read_text(
        encoding="utf-8"
    )

    assert config["status"] == "secondary_descriptive"
    assert config["source"]["arxiv"] == "2607.01233"
    assert config["bootstrap"] == {
        "resamples": 10000,
        "seed": 17,
        "confidence_level": 0.95,
    }
    assert "EmbeddingCache" in script
    assert "embed_texts" not in script
    assert "cache.get_array" in script
    assert "vectors: dict" not in script
    assert "summarize_same_prompt_geometry_metrics" in script
    assert "cached embedding is missing" in script
    assert "scripts/analyze_same_prompt_geometry.py" in slurm
    assert "same-prompt-geometry-secondary.json" in slurm
    assert "same-prompt-geometry-findings.md" in slurm
    assert "scripts/analyze_same_prompt_geometry.py" in exposure_slurm
    assert "same-prompt-geometry-secondary.json" in exposure_slurm
    assert "scripts/analyze_same_prompt_geometry.py" in promoted_slurm
    assert "same-prompt-geometry-secondary.json" in promoted_slurm
    assert "scripts/analyze_same_prompt_geometry.py" in drkl_slurm
    assert "same-prompt-geometry-secondary.json" in drkl_slurm


def test_same_prompt_geometry_cli_reads_only_validated_cached_embeddings(tmp_path: Path) -> None:
    instruction = "Represent the idea."
    annotation = tmp_path / "annotation.yaml"
    annotation.write_text(
        yaml.safe_dump(
            {
                "embedding_model": "embedding-model",
                "embedding_revision": "e" * 40,
                "embedding_instruction": instruction,
                "embedding_batch_size": 8,
                "embedding_max_length": 2048,
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    cache = EmbeddingCache(
        cache_root,
        model_id="embedding-model",
        revision="e" * 40,
        max_length=2048,
        batch_size=8,
    )
    method_vectors = {
        "A1": (("teacher", (1.0, 0.0)),),
        "A3": (("human", (0.0, 1.0)),),
        "A0": (("base-1", (1.0, 0.0)), ("base-2", (2**-0.5, 2**-0.5))),
        "B1": (("trained-1", (0.0, 1.0)), ("trained-2", (0.0, 1.0))),
    }
    score_dirs: dict[str, Path] = {}
    for method, samples in method_vectors.items():
        directory = tmp_path / method
        directory.mkdir()
        score_dirs[method] = directory
        for prompt_id in ("p1", "p2"):
            texts = [f"{text}-{prompt_id}" for text, _ in samples]
            for text, (_, vector) in zip(texts, samples, strict=True):
                cache.put(f"Instruct: {instruction}\nQuery: {text}", vector)
            records = [
                {
                    "prompt_id": prompt_id,
                    "sample_index": index,
                    "text": text,
                    "finish_reason": "stop",
                    "completion_tokens": 4,
                    "dimensions": {
                        "relevance": 5,
                        "feasibility": 5,
                        "soundness": 5,
                        "clarity": 5,
                        "instruction_compliance": 5,
                    },
                    "quality_score": 1.0,
                    "request_id": f"{method}-{prompt_id}-{index}",
                    "model": "judge",
                }
                for index, text in enumerate(texts)
            ]
            (directory / f"{prompt_id}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "prompt_id": prompt_id,
                        "text_hashes": [
                            hashlib.sha256(text.encode()).hexdigest() for text in texts
                        ],
                        "judge": {
                            "model": "judge",
                            "revision": "j" * 40,
                            "max_tokens": 128,
                        },
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )

    output = tmp_path / "result.json"
    markdown = tmp_path / "result.md"
    command = [sys.executable, "scripts/analyze_same_prompt_geometry.py"]
    for method, directory in score_dirs.items():
        command.extend(("--input", f"{method}={directory}"))
    command.extend(
        (
            "--embedding-cache-dir",
            str(cache_root),
            "--annotation-config",
            str(annotation),
            "--analysis-config",
            "configs/evaluation/same_prompt_geometry.yaml",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        )
    )
    subprocess.run(command, check=True)

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["num_prompts"] == 2
    assert result["methods"]["B1"]["metrics"]["human_cosine_mean"]["estimate"] == 1
    assert "Positive teacher-minus-human affinity" in markdown.read_text(encoding="utf-8")


def test_same_prompt_geometry_reuses_identical_bootstrap_draws_across_methods() -> None:
    names = (
        "teacher_cosine_mean",
        "human_cosine_mean",
        "teacher_minus_human_affinity",
        "within_method_pair_cosine",
    )
    prompts = [f"p{index}" for index in range(9)]
    base = {
        prompt: dict(zip(names, (0.6 + index / 100, 0.5, 0.1, 0.7), strict=True))
        for index, prompt in enumerate(prompts)
    }
    treatment = {
        prompt: dict(zip(names, (0.7, 0.4 + index / 100, 0.3, 0.8), strict=True))
        for index, prompt in enumerate(prompts)
    }
    result = summarize_same_prompt_geometry_metrics(
        {"A0": base, "B1": treatment, "B2": treatment},
        samples_per_prompt={"A0": [16] * 9, "B1": [16] * 9, "B2": [16] * 9},
        reference_by_prompt={prompt: 0.55 for prompt in prompts},
        human_method="A3",
        teacher_method="A1",
        base_method="A0",
        embedding_dimension=2560,
        resamples=11,
        seed=17,
    )

    assert result["methods"]["B1"]["metrics"] == result["methods"]["B2"]["metrics"]
