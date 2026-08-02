"""Pinned-transformers embedding helper shared by calibration and model evaluation."""

from collections.abc import Sequence


def embed_texts(
    texts: Sequence[str],
    *,
    model_id: str,
    revision: str,
    max_length: int,
    batch_size: int,
) -> list[list[float]]:
    """Return normalized last-token embeddings from the official Qwen embedding model."""

    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    if not texts:
        raise ValueError("embedding input must be non-empty")
    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
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
    embeddings: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)
            hidden = model(**batch).last_hidden_state
            pooled = hidden[:, -1]
            normalized = functional.normalize(pooled.float(), p=2, dim=1)
            embeddings.extend(normalized.cpu().tolist())
    return embeddings
