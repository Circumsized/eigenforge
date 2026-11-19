"""Durable, atomic file replacement shared by checkpoint and artifact manifests.

``os.replace`` makes the *namespace* swap atomic (a reader sees either the old
file or the new one, never a half-written file), but it does **not** make the
write durable. Data and rename metadata can still be sitting in the page cache /
disk write-back cache, so a crash or power loss shortly after ``os.replace`` can
surface a zero-length or partial file under the target name. On ext4 with
delayed allocation the rename metadata may even be journaled *before* the data
blocks are written, which is exactly the "present but empty file" failure mode.

The standard remedy is to ``fsync`` the file's data before the rename and to
``fsync`` the parent directory afterwards so the rename itself is durable. The
directory fd is opened *before* the rename: a directory fd only observes
writeback errors that happen after it was opened (LWN 752063, "fsync gates"),
so opening it after the swap could miss a rename error and falsely report the
swap as durable. On Windows there is no supported directory fsync (``os.open``
on a directory raises ``PermissionError`` and ``FlushFileBuffers`` on a
directory handle is denied), so that step is skipped there: the file-data fsync
(``os.fsync`` maps to ``_commit``/``FlushFileBuffers``) makes the *data*
durable, while the namespace swap itself is best-effort because ``os.replace``
passes only ``MOVEFILE_REPLACE_EXISTING``, not ``MOVEFILE_WRITE_THROUGH`` — the
gap the atomicwrites library closes by calling ``MoveFileExW`` directly.

References:
- CPython ``os.fsync`` / ``os.replace`` docs (flush-then-fsync guidance).
- atomicwrites (fsync file + parent directory rationale; MoveFileExW on Windows).
- LWN "Delayed allocation and the zero-length file problem" (Ts'o).
- LWN 752063, "Ensure that fsync() errors are not ignored" (fsync gates).
- Dan Luu, "Files are hard".
- Calvin Loncaric, "How to Durably Write a File on POSIX Systems".
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import IO, Any, Callable

from eigenforge.common.logger import logger


def _open_directory(directory: Path) -> int | None:
    """Open a directory fd for a later fsync (POSIX only; ``None`` elsewhere).

    Windows cannot open or flush a directory handle through the standard
    library, so this returns ``None`` there; ``durable_replace`` is still
    namespace-atomic on Windows via ``os.replace``. A POSIX open failure is
    logged: the caller would otherwise silently lose the rename-durability
    guarantee while believing it holds it.
    """
    if sys.platform == "win32":
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        return os.open(directory, flags)
    except OSError as exc:
        logger.warning(
            "Cannot open directory {} for fsync ({}); the rename will be atomic "
            "but its durability is unverified.",
            directory, exc,
        )
        return None


def durable_replace(path: str | os.PathLike[str], write: Callable[[IO[str]], None]) -> None:
    """Atomically and (on POSIX) durably create/replace ``path``.

    ``write`` receives an open text handle positioned at the start of a temp file
    in the *same directory* as ``path`` (guaranteeing the final rename stays
    within one filesystem, which is what makes it atomic). The temp file's data
    is flushed and fsynced, then swapped into place, then the parent directory
    is fsynced on POSIX. On any failure the temp file is removed and the original
    ``path`` is left untouched.
    """
    target = Path(path)
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    dir_fd = _open_directory(directory)
    tmp_name: str | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{target.name}-", suffix=".tmp")
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8", newline="")
        except BaseException:
            # fdopen itself failed: the raw descriptor must not leak.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        with handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError as exc:
                # Some filesystems refuse fsync on a directory; the rename is
                # still atomic, so degrade quietly rather than failing a write
                # that has already succeeded.
                logger.warning(
                    "Directory fsync after replace failed ({}); the swap is "
                    "atomic but its durability is unverified.",
                    exc,
                )
    except BaseException:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError as exc:
                logger.warning(
                    "Temp file '{}' could not be removed ({}); stale residue remains.",
                    tmp_name, exc,
                )
        raise
    finally:
        if dir_fd is not None:
            try:
                os.close(dir_fd)
            except OSError as exc:
                # A close failure here means the directory fd outlives the
                # function; on NFS a deferred write error surfaces exactly at
                # close(EIO), so it must not pass silently.
                logger.warning("Directory fd close failed after replace ({});", exc)


def durable_replace_bytes(path: str | os.PathLike[str], write: Callable[[IO[bytes]], None]) -> None:
    """Binary twin of :func:`durable_replace` for payloads that are not text.

    The write-order contract is identical (data fsync -> atomic rename ->
    POSIX directory fsync; temp removed and the original left untouched on any
    failure). ``torch.save`` accepts a file-like object, so model checkpoints
    get the same durability guarantee as the text manifests instead of being
    written straight onto their final path (where a crash truncates the only
    copy of ``latest.pt``).
    """
    target = Path(path)
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    dir_fd = _open_directory(directory)
    tmp_name: str | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{target.name}-", suffix=".tmp")
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            # fdopen itself failed: the raw descriptor must not leak.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        with handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError as exc:
                logger.warning(
                    "Directory fsync after replace failed ({}); the swap is "
                    "atomic but its durability is unverified.",
                    exc,
                )
    except BaseException:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError as exc:
                logger.warning(
                    "Temp file '{}' could not be removed ({}); stale residue remains.",
                    tmp_name, exc,
                )
        raise
    finally:
        if dir_fd is not None:
            try:
                os.close(dir_fd)
            except OSError as exc:
                # A close failure here means the directory fd outlives the
                # function; on NFS a deferred write error surfaces exactly at
                # close(EIO), so it must not pass silently.
                logger.warning("Directory fd close failed after replace ({});", exc)


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Write ``payload`` as indented JSON through :func:`durable_replace`.

    Machine-generated reports (smoke, quality gate) are not authoritative
    pointers, but a half-written JSON report is still an unreadable report, so
    they converge on the same durable-write primitive as the authoritative
    manifests.
    """
    durable_replace(
        path,
        lambda handle: json.dump(payload, handle, ensure_ascii=False, indent=2),
    )
