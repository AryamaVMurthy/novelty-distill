"""Fail-closed completion checks for resumable adapter training jobs."""

import json
from pathlib import Path


def adapter_run_complete(
    metadata_path: Path, *, baseline_id: str, max_steps: int
) -> bool:
    """Return false for a missing run and validate a claimed completed LoRA adapter."""

    if not metadata_path.exists():
        return False
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("baseline_id") != baseline_id or payload.get("max_steps") != max_steps:
        raise ValueError(f"stale or incompatible training metadata {metadata_path}")
    raw_final_dir = payload.get("final_dir")
    if not isinstance(raw_final_dir, str) or not raw_final_dir:
        raise ValueError(f"training metadata has no final adapter directory: {metadata_path}")
    final_dir = Path(raw_final_dir).resolve()
    output_dir = metadata_path.parent.resolve()
    if not final_dir.is_relative_to(output_dir):
        raise ValueError(f"final adapter escapes its output directory: {final_dir}")
    required = (final_dir / "adapter_config.json", final_dir / "adapter_model.safetensors")
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError(
            f"completed training metadata points to an incomplete adapter: {final_dir}"
        )
    return True
