"""Contract for the shared durable atomic-replace helper (common/fsutil.py).

``durable_replace`` must (a) publish the new content atomically, (b) leave the
target untouched if the write raises, and (c) not leave temp files behind.
"""

from __future__ import annotations

import os

import pytest

from eigenforge.common.fsutil import durable_replace


def test_replaces_existing_file_content(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("old", encoding="utf-8")
    durable_replace(target, lambda h: h.write("new"))
    assert target.read_text(encoding="utf-8") == "new"


def test_creates_file_and_missing_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "data.txt"
    durable_replace(target, lambda h: h.write("hello"))
    assert target.read_text(encoding="utf-8") == "hello"


def test_failed_write_keeps_previous_content_and_cleans_temp(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("original", encoding="utf-8")

    def boom(handle):
        handle.write("partial")
        raise OSError("disk full")

    with pytest.raises(OSError):
        durable_replace(target, boom)

    assert target.read_text(encoding="utf-8") == "original"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "data.txt"]
    assert leftovers == [], f"temp files must be cleaned up, found {leftovers}"


def test_fsync_is_invoked_on_the_written_file(tmp_path, monkeypatch):
    """The durability step must actually call fsync on the temp file's data."""
    calls: list[int] = []
    real_fsync = os.fsync

    def spy_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    target = tmp_path / "data.txt"
    durable_replace(target, lambda h: h.write("durable"))
    assert calls, "durable_replace must fsync before publishing"


def test_failed_replace_keeps_previous_content_and_cleans_temp(tmp_path, monkeypatch):
    """``os.replace`` can fail on real platforms (a Windows sharing violation:
    a backup/AV scanner holding the target open without FILE_SHARE_DELETE). The
    target must survive untouched and no temp residue may remain."""
    target = tmp_path / "data.txt"
    target.write_text("original", encoding="utf-8")

    def broken_replace(src, dst):
        raise OSError("target locked")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError):
        durable_replace(target, lambda h: h.write("new"))

    assert target.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


def test_fsync_failure_propagates_and_keeps_previous_content(tmp_path, monkeypatch):
    """fsyncgate: a failing fsync must FAIL the write (never report success) —
    the temp file is removed and the target keeps its previous content."""
    target = tmp_path / "data.txt"
    target.write_text("original", encoding="utf-8")

    def broken_fsync(fd):
        raise OSError("EIO during fsync")

    monkeypatch.setattr(os, "fsync", broken_fsync)
    with pytest.raises(OSError):
        durable_replace(target, lambda h: h.write("new"))

    assert target.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


def test_fsync_precedes_replace_and_directory_fsync_follows(tmp_path, monkeypatch):
    """Order contract: the data fsync must happen BEFORE the rename (otherwise
    a crash can surface a renamed-but-unwritten file) and, on POSIX, the
    directory fsync must follow the rename (fsync-gates semantics)."""
    import sys

    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def spy_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    target = tmp_path / "data.txt"
    durable_replace(target, lambda h: h.write("ordered"))

    assert events[0] == "fsync" and events[1] == "replace", events
    if sys.platform != "win32":
        # POSIX: data fsync, rename, then the directory fsync.
        assert events == ["fsync", "replace", "fsync"], events
    else:
        # Windows has no directory fsync; exactly one fsync is expected.
        assert events == ["fsync", "replace"], events


def test_text_mode_writes_lf_without_platform_translation(tmp_path):
    """``newline=""`` is load-bearing: default text mode would translate ``\\n``
    to ``os.linesep`` on Windows, silently rewriting the authoritative bytes of
    the manifest on every persisted write."""
    target = tmp_path / "lines.txt"
    durable_replace(target, lambda h: h.write("a\nb\n"))
    assert target.read_bytes() == b"a\nb\n"
