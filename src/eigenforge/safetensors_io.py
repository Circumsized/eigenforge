"""Optional safetensors weight-only interchange for trusted checkpoints.

This module intentionally stores only generator/EMA tensor state dictionaries.
Optimizer state, epoch, training configuration and provenance remain in the
existing JSON/manifest checkpoint contract.  It is an optional export path,
not a replacement for ``checkpoints.py`` and never changes the default trainer.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping

from eigenforge.common.fsutil import durable_replace_bytes


def _require_backend():
    if importlib.util.find_spec("safetensors") is None:
        raise RuntimeError("safetensors is not configured; install the optional 'safetensors' extra")
    from safetensors.torch import load_file, save

    return load_file, save


def export_state_dict(
    state_dict: Mapping[str, Any],
    path: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write a tensor-only state dict in the safe, non-pickle format.

    The payload is serialized in memory and swapped into place through
    ``fsutil.durable_replace_bytes`` (data fsync -> atomic rename -> POSIX
    directory fsync) — the same durability contract as the torch checkpoints.
    Writing straight onto the final path (what ``save_file`` does) leaves a
    truncated .safetensors under the real name if the process dies mid-write.
    """
    _, save = _require_backend()
    import torch

    tensors = {}
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"safetensors accepts tensors only; {name!r} is {type(value).__name__}")
        tensors[str(name)] = value.detach().cpu().contiguous()
    target = Path(path)
    safe_metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
    payload = save(tensors, metadata=safe_metadata)
    durable_replace_bytes(target, lambda handle: handle.write(payload))
    return target


def import_state_dict(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load tensor-only weights; no pickle execution path is involved."""
    load_file, _ = _require_backend()
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"safetensors file does not exist: {target}")
    return dict(load_file(str(target), device=device))


def metadata(path: str | Path) -> dict[str, str]:
    """Return only safetensors header metadata for audit/reporting."""
    if importlib.util.find_spec("safetensors") is None:
        raise RuntimeError("safetensors is not configured; install the optional 'safetensors' extra")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})
