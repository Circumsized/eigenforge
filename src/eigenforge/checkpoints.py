"""Checkpoint save/load contract: durability, traceability, and verification.

Properties this module gives both training entry points (they used to each
hand-roll them, and both hand-rolls were weaker):

1. **Durability** — checkpoints go through ``fsutil.durable_replace_bytes``
   (same-directory tmp -> fsync -> atomic rename -> directory fsync), so an
   interrupted epoch can no longer truncate the only copy of ``latest.pt``.
2. **Traceability** — every save writes a sibling ``<artifact>.manifest.json``
   produced by :class:`eigenforge.artifacts.ArtifactManifest`
   (sha256 / config / seed / step / git).
3. **Verification** — resume checks the sibling manifest's sha256 against the
   file bytes and the recorded training config against the current CLI
   arguments (architecture anti-drift), and requires a strict key/epoch
   schema instead of silently defaulting ``epoch`` to 0. The bytes that were
   hashed are the very bytes handed to ``torch.load`` (read once), so a file
   swapped between "verified" and "deserialized" cannot ride a stale hash.
4. **Crash recovery** — two files cannot be replaced atomically together, so
   the previous verified generation is kept as ``<artifact>.prev`` (+ its own
   manifest) via hard links *before* the new pair is written. Everything that
   can fail without touching disk (serialization, hashing, manifest JSON)
   runs first. A crash between the checkpoint rename and the manifest write
   leaves a new checkpoint next to an old manifest; resume with
   ``fallback_to_previous=True`` then loads the intact previous generation
   (loud warning) instead of dead-ending. This is the commit protocol used by
   crash-consistent checkpointers: data first, hash-bearing manifest last,
   keep the last known-good generation, fall back newest -> oldest.

Backward compatibility: a checkpoint without a sibling manifest loads with a
warning (the manifest mechanism was introduced later; historical files get
no fabricated hash) unless ``require_manifest`` is set — the training entry
points require it by default. Missing ``ema_*`` keys fall back to the raw
weights **only** in ``train_eigengan`` (loud warning, commit d9f7df4's
semantics); ``train_diffusion`` lists ``ema_model`` as required, so a
checkpoint without it is a hard schema error there (fail-closed, no
pseudo-EMA).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from eigenforge.artifacts import ArtifactManifest
from eigenforge.common.fsutil import durable_replace_bytes
from eigenforge.common.logger import logger
from eigenforge.common.torch_guard import require_safe_torch

_PREV_SUFFIX = ".prev"


class CheckpointError(RuntimeError):
    """Base class for checkpoint contract violations."""


class CheckpointIntegrityError(CheckpointError):
    """Checkpoint bytes do not match their manifest (corruption or tampering)."""


class CheckpointSchemaError(CheckpointError):
    """Checkpoint payload is missing required structure or disagrees with config."""


class NonFiniteLossError(CheckpointError):
    """A monitored loss/parameter went NaN/Inf; the run must stop instead of saving junk."""


def _comparable(value: Any) -> Any:
    """JSON-normalize a config value so manifests compare equal to callers.

    A manifest round-trips through JSON (tuple -> list); without this a caller
    passing the identical-looking tuple would trip a phantom drift error.
    Non-serializable values fall through unchanged.
    """
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return value


def _manifest_path(artifact_path: Path) -> Path:
    return Path(f"{artifact_path}.manifest.json")


def _prev_path(artifact_path: Path) -> Path:
    return Path(f"{artifact_path}{_PREV_SUFFIX}")


@dataclass(frozen=True)
class ResumeState:
    """A verified checkpoint ready for entry-point-specific wiring."""

    payload: dict[str, Any]
    epoch: int
    manifest: dict[str, Any] | None


def _retain_previous_generation(target: Path) -> None:
    """Hard-link the current checkpoint+manifest to ``<name>.prev`` (best effort).

    A hard link costs no extra disk or copy time, and the later atomic replace
    of ``target`` leaves the old inode alive under the ``.prev`` name. Each
    link is staged and renamed over the destination so ``.prev`` is never
    half-updated. Filesystems without hard links degrade (loudly) to the
    pre-existing single-generation behavior rather than failing the save.
    """
    prev = _prev_path(target)
    for source, destination in ((target, prev), (_manifest_path(target), _manifest_path(prev))):
        if not source.is_file():
            continue
        staging = destination.with_name(f".{destination.name}.link")
        try:
            staging.unlink(missing_ok=True)
            os.link(source, staging)
            os.replace(staging, destination)
        except OSError as exc:
            logger.warning(
                "Could not retain the previous checkpoint generation for {} ({}); "
                "an interrupted save will not be recoverable from {}.",
                target, exc, prev.name,
            )
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass
            return


def save_checkpoint(
    artifact_path: str | Path,
    payload: Mapping[str, Any],
    *,
    model_kind: str,
    seed: int,
    step: int,
    training_config: Mapping[str, Any],
    dataset: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> ArtifactManifest:
    """Durably write ``payload`` and its sibling manifest; return the manifest."""
    target = Path(artifact_path)

    # Phase 1 — nothing below touches disk, so any failure (unserializable
    # payload, non-JSON config value, bad seed/step) leaves the previous
    # checkpoint+manifest pair exactly as it was.
    # ponytail: serializing to memory doubles peak RAM for one checkpoint;
    # fine for these 64px models. Upgrade path for multi-GB states: stream to
    # a tmp file and hash in chunks.
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    data = buffer.getvalue()
    manifest = ArtifactManifest.create(
        artifact_path=target,
        model_kind=model_kind,
        seed=seed,
        step=step,
        training_config=dict(training_config),
        dataset=dict(dataset) if dataset else None,
        metrics=dict(metrics) if metrics else None,
        artifact_sha256=hashlib.sha256(data).hexdigest(),
    )
    manifest.to_json_text()

    # Phase 2 — keep the last known-good generation, then data first,
    # hash-bearing manifest last.
    _retain_previous_generation(target)
    durable_replace_bytes(target, lambda handle: handle.write(data))
    manifest.to_json(_manifest_path(target))
    return manifest


def _load_verified(
    target: Path,
    *,
    map_location: Any,
    required_keys: Sequence[str],
    expected_config: Mapping[str, Any] | None,
    require_manifest: bool,
    require_epoch: bool,
) -> ResumeState:
    if not target.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {target}")

    manifest: dict[str, Any] | None = None
    sidecar = _manifest_path(target)
    if sidecar.exists() and not sidecar.is_file():
        raise CheckpointIntegrityError(
            f"manifest path {sidecar} exists but is not a regular file; refusing to resume"
        )
    loaded: dict[str, Any] | None = None
    if sidecar.is_file():
        try:
            parsed = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointIntegrityError(
                f"manifest {sidecar} is unreadable ({exc}); refusing to resume "
                "from an unverifiable checkpoint"
            ) from exc
        if not isinstance(parsed, dict):
            raise CheckpointIntegrityError(
                f"manifest {sidecar} is not a JSON object; refusing to resume"
            )
        loaded = parsed
    elif require_manifest:
        raise CheckpointIntegrityError(
            f"checkpoint {target} has no sibling manifest; refusing to resume "
            "without verifiable provenance (require_manifest=True; the training "
            "CLIs accept --allow-unverified-resume to override for legacy files)"
        )

    # Single read: hash and deserialize the SAME bytes. Opening the path twice
    # (hash, then torch.load(path)) lets a file swapped in between ride a hash
    # that no longer describes what is deserialized.
    data = target.read_bytes()

    if loaded is not None:
        actual = hashlib.sha256(data).hexdigest()
        if loaded.get("artifact_sha256") != actual:
            raise CheckpointIntegrityError(
                f"checkpoint {target} does not match its manifest sha256 "
                f"(file={actual[:16]}…, manifest={str(loaded.get('artifact_sha256'))[:16]}…); "
                "the artifact is corrupt or was replaced after the manifest was written"
            )
        manifest = loaded
        if expected_config:
            recorded = manifest.get("training_config") or {}
            drift = {
                key: (recorded.get(key), value)
                for key, value in expected_config.items()
                # JSON round-trip normalization: manifests store lists where
                # callers may pass tuples; comparing raw would report phantom
                # drift for perfectly matching architecture.
                if _comparable(recorded.get(key)) != _comparable(value)
            }
            if drift:
                raise CheckpointSchemaError(
                    "checkpoint architecture disagrees with this run "
                    f"(recorded vs requested): {drift}; train a fresh model or "
                    "pass matching arguments"
                )
    else:
        logger.warning(
            "Checkpoint {} has no sibling manifest; resuming without sha256/config verification.",
            target,
        )

    payload = torch.load(io.BytesIO(data), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise CheckpointSchemaError(
            f"checkpoint {target} does not contain a state mapping (got {type(payload).__name__})"
        )

    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise CheckpointSchemaError(
            f"checkpoint {target} is missing required keys {missing}; this entry "
            "point can only resume checkpoints it produced itself"
        )
    if "epoch" not in payload:
        if require_epoch:
            raise CheckpointSchemaError(
                f"checkpoint {target} lacks 'epoch'; refusing to silently restart from 0"
            )
        return ResumeState(payload=payload, epoch=0, manifest=manifest)
    epoch = payload["epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise CheckpointSchemaError(
            f"checkpoint {target} has invalid 'epoch': {epoch!r}"
        )
    return ResumeState(payload=payload, epoch=epoch, manifest=manifest)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    required_keys: Sequence[str] = (),
    expected_config: Mapping[str, Any] | None = None,
    require_manifest: bool = False,
    require_epoch: bool = True,
    fallback_to_previous: bool = False,
) -> ResumeState:
    """Load a checkpoint with manifest verification and a strict schema.

    ``require_manifest`` refuses to resume without the sibling manifest — the
    strict mode for accredited training pipelines; the default keeps the
    historical no-manifest path working (loud warning, no fabricated hash).

    ``require_epoch=False`` is for read-only consumers (e.g. latent traversal)
    that accept a bare state mapping; resume paths keep the default.

    ``fallback_to_previous`` retries the ``<name>.prev`` generation when the
    primary file is missing or fails integrity verification (an interrupted
    save). Schema/config errors never fall back: a drifted architecture is the
    operator's mistake, not corruption, and the previous generation would
    drift identically.

    Raises :class:`CheckpointIntegrityError` for absent (when required),
    corrupt, non-regular-file manifests and sha mismatches, and
    :class:`CheckpointSchemaError` for missing keys, a missing/non-integer
    ``epoch``, or a manifest config that disagrees with ``expected_config``.
    """
    target = Path(path)
    require_safe_torch(when="load_checkpoint")
    options = {
        "map_location": map_location,
        "required_keys": required_keys,
        "expected_config": expected_config,
        "require_manifest": require_manifest,
        "require_epoch": require_epoch,
    }
    try:
        return _load_verified(target, **options)
    except (CheckpointIntegrityError, FileNotFoundError) as primary:
        previous = _prev_path(target)
        if not fallback_to_previous or not previous.is_file():
            raise
        logger.warning(
            "Checkpoint {} is unusable ({}); falling back to the previous generation {}.",
            target, primary, previous,
        )
        try:
            return _load_verified(previous, **options)
        except CheckpointError as secondary:
            raise primary from secondary


def select_generator_state(payload: Mapping[str, Any], *, source: str) -> Mapping[str, Any]:
    """Pick the generator weights a read-only consumer should render with.

    EMA weights are what training samples with; falling back to the raw
    generator (or a bare state mapping) is legitimate for older files but must
    never be silent — otherwise every grid rendered from it is quietly a
    different model than the one the training logs describe.
    """
    if "ema_generator" in payload:
        return payload["ema_generator"]
    if "generator" in payload:
        logger.warning(
            "Checkpoint {} lacks ema_generator; using the RAW generator weights (not the EMA used for sampling).",
            source,
        )
        return payload["generator"]
    logger.warning(
        "Checkpoint {} has neither ema_generator nor generator; treating it as a bare generator state_dict.",
        source,
    )
    return payload


def check_finite_losses(
    losses: Mapping[str, "torch.Tensor"], *, epoch: int, step: int
) -> None:
    """Raise :class:`NonFiniteLossError` if any monitored loss is NaN/Inf.

    Called *before* the optimizer steps consume the losses so a non-finite
    iteration never reaches the weights. Raising (not ``assert``) keeps this
    live under ``python -O``. The caller must not save this step's
    checkpoint: with the atomic writer, the on-disk ``latest.pt`` plus
    manifest remain the previous intact epoch's pair.
    """
    exploded = {
        name: float(value.detach().item())
        for name, value in losses.items()
        if not bool(torch.isfinite(value).all())
    }
    if exploded:
        logger.error(
            "Non-finite loss at epoch={} step={}: {}; stopping before saving a corrupt checkpoint.",
            epoch, step, exploded,
        )
        raise NonFiniteLossError(
            f"non-finite losses at epoch={epoch} step={step}: {exploded}"
        )


def check_finite_parameters(
    modules: Mapping[str, "torch.nn.Module"],
    *,
    epoch: int,
    context: str = "at end of",
) -> None:
    """Raise :class:`NonFiniteLossError` if any floating-point state is NaN/Inf.

    The loss breaker only inspects values *before* an update; the final
    optimizer step of an epoch (or an AMP step that kept the loss finite while
    corrupting weights) would otherwise be saved with a perfectly valid
    manifest. Call this immediately before :func:`save_checkpoint`; the preview
    path calls it mid-epoch. ``context`` only shapes the error message.
    """
    corrupted = [
        f"{module_name}.{tensor_name}"
        for module_name, module in modules.items()
        for tensor_name, tensor in module.state_dict().items()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
    ]
    if corrupted:
        logger.error(
            "Non-finite parameters {} epoch={}: {}; refusing to save.",
            context, epoch, corrupted[:8],
        )
        raise NonFiniteLossError(
            f"non-finite parameters {context} epoch={epoch}: {corrupted[:8]}"
            + (f" (+{len(corrupted) - 8} more)" if len(corrupted) > 8 else "")
        )
