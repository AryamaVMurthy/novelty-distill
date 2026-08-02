import json

import numpy as np
import pytest

from novelty_distill.evaluation.embeddings import EmbeddingCache, embed_texts


def _cache(tmp_path) -> EmbeddingCache:
    return EmbeddingCache(
        tmp_path,
        model_id="embedding-model",
        revision="a" * 40,
        max_length=128,
        batch_size=8,
    )


def test_embedding_cache_round_trips_normalized_float32_vectors(tmp_path) -> None:
    cache = _cache(tmp_path)
    assert cache.get("alpha") is None

    cache.put("alpha", (0.6, 0.8))

    assert cache.get("alpha") == pytest.approx([0.6, 0.8])
    manifest = json.loads((cache.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint"] == cache.fingerprint
    assert manifest["controls"]["batch_size"] == 8


def test_embedding_cache_rejects_corrupt_or_unnormalized_vectors(tmp_path) -> None:
    cache = _cache(tmp_path)
    with pytest.raises(ValueError, match="not normalized"):
        cache.put("bad", (1.0, 1.0))

    path = cache._path("corrupt")
    path.parent.mkdir(parents=True)
    np.save(path, np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="shape or dtype"):
        cache.get("corrupt")


def test_embed_texts_uses_complete_cache_without_loading_a_model(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.put("alpha", (0.6, 0.8))

    embedded = embed_texts(
        ("alpha", "alpha"),
        model_id="embedding-model",
        revision="a" * 40,
        max_length=128,
        batch_size=8,
        cache_dir=tmp_path,
    )

    assert np.allclose(embedded, [[0.6, 0.8], [0.6, 0.8]])
