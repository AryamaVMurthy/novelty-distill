"""Pinned-transformers embeddings with a resumable content-addressed cache."""

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

EMBEDDING_ALGORITHM = "qwen-last-token-l2-v1"


def embedding_cache_fingerprint(
    *, model_id: str, revision: str, max_length: int, batch_size: int
) -> str:
    """Fingerprint every control that can affect cached embedding values."""

    controls = {
        "algorithm": EMBEDDING_ALGORITHM,
        "attention_implementation": "sdpa",
        "batch_size": batch_size,
        "dtype": "bfloat16",
        "max_length": max_length,
        "model": model_id,
        "revision": revision,
    }
    return hashlib.sha256(
        json.dumps(controls, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class EmbeddingCache:
    """Atomic per-text float32 cache scoped to one exact embedding configuration."""

    def __init__(
        self,
        root: Path,
        *,
        model_id: str,
        revision: str,
        max_length: int,
        batch_size: int,
    ) -> None:
        self.controls = {
            "algorithm": EMBEDDING_ALGORITHM,
            "attention_implementation": "sdpa",
            "batch_size": batch_size,
            "dtype": "bfloat16",
            "max_length": max_length,
            "model": model_id,
            "revision": revision,
        }
        self.fingerprint = embedding_cache_fingerprint(
            model_id=model_id,
            revision=revision,
            max_length=max_length,
            batch_size=batch_size,
        )
        self.directory = root / self.fingerprint
        self._ensure_manifest()

    def get(self, text: str) -> list[float] | None:
        path = self._path(text)
        if not path.exists():
            return None
        try:
            vector = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid cached embedding {path}") from error
        self._validate_vector(vector, path=path)
        return vector.tolist()

    def put(self, text: str, embedding: Sequence[float]) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        path = self._path(text)
        self._validate_vector(vector, path=path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                np.save(handle, vector, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return self.directory / digest[:2] / f"{digest}.npy"

    def _ensure_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "fingerprint": self.fingerprint,
            "controls": self.controls,
        }
        path = self.directory / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"embedding cache manifest changed: {path}")
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".manifest.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(manifest, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != manifest:
                    raise ValueError(f"embedding cache manifest changed: {path}")
            else:
                os.replace(temporary_name, path)
                temporary_name = None
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _validate_vector(vector: np.ndarray, *, path: Path) -> None:
        if vector.dtype != np.float32 or vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"cached embedding has an invalid shape or dtype: {path}")
        if not np.isfinite(vector).all():
            raise ValueError(f"cached embedding contains non-finite values: {path}")
        norm = float(np.linalg.norm(vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"cached embedding is not normalized: {path}")


def embed_texts(
    texts: Sequence[str],
    *,
    model_id: str,
    revision: str,
    max_length: int,
    batch_size: int,
    cache_dir: Path | None = None,
) -> list[list[float]]:
    """Return normalized last-token embeddings, atomically caching each completed text."""

    if not texts:
        raise ValueError("embedding input must be non-empty")
    if any(not text for text in texts):
        raise ValueError("embedding texts must be non-empty")
    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")

    cache = (
        EmbeddingCache(
            cache_dir,
            model_id=model_id,
            revision=revision,
            max_length=max_length,
            batch_size=batch_size,
        )
        if cache_dir is not None
        else None
    )
    by_text: dict[str, list[float]] = {}
    missing: list[str] = []
    for text in dict.fromkeys(texts):
        cached = cache.get(text) if cache is not None else None
        if cached is None:
            missing.append(text)
        else:
            by_text[text] = cached
    cache_hits = len(by_text)

    if missing:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        model = (
            AutoModel.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .cuda()
            .eval()
        )
        with torch.inference_mode():
            for start in range(0, len(missing), batch_size):
                batch_texts = missing[start : start + batch_size]
                batch = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(model.device)
                hidden = model(**batch).last_hidden_state
                pooled = hidden[:, -1]
                normalized = functional.normalize(pooled.float(), p=2, dim=1)
                batch_embeddings = normalized.cpu().tolist()
                for text, embedding in zip(batch_texts, batch_embeddings, strict=True):
                    if cache is not None:
                        cache.put(text, embedding)
                    by_text[text] = embedding
                print(
                    json.dumps(
                        {
                            "embedding_cache_hits": cache_hits,
                            "embedded": min(start + len(batch_texts), len(missing)),
                            "missing_at_start": len(missing),
                        },
                        sort_keys=True,
                    )
                )
    return [by_text[text] for text in texts]
