"""torch.load safety floor for checkpoint loading.

Two advisories define the floor (both are "``weights_only=True`` is not a
boundary on this version" bugs, so the flag alone never immunizes an old torch):

* GHSA-53q9-r3pm-6pq6 / CVE-2025-32434 (CVSS v4 9.3, critical): in PyTorch
  <= 2.5.1 even ``torch.load(..., weights_only=True)`` was exploitable for
  remote code execution; fixed in 2.6.0.
* GHSA-63cw-57p8-fm3p / CVE-2026-24747 (high, published 2026-01-27): the
  ``weights_only`` unpickler corrupts memory on crafted checkpoints, allowing
  arbitrary code execution; affects ``< 2.10.0`` (repo advisory: ``<= 2.9.1``),
  fixed in 2.10.0. Raising the floor only to 2.6.0 therefore still admits an
  exploitable runtime.

The repo forbids ``torch.load`` without ``weights_only=True`` (AGENTS.md
§3-3), and pip never upgrades an already-installed version that merely
satisfies an old ``torch>=…`` floor — so the floor is also checked at runtime
here.

The check is an explicit ``raise`` (never an ``assert``): ``python -O``
strips asserts, which would silently turn this guard into a no-op exactly
like the loss-finiteness checks in ``smoke.py``.

The escape hatch reads ``EIGENFORGE_ALLOW_UNSAFE_TORCH`` from the *process
environment* only (``os.environ``), not from ``.env``: it must be exported
explicitly so it cannot be enabled by a stray dotfile. It is read directly
from ``os.environ`` rather than through ``AppConfig`` so the startup guard
remains self-contained and never depends on dotfile parsing.
"""

from __future__ import annotations

import os
import re
import sys

import torch

from eigenforge.common.logger import logger

MIN_SAFE_TORCH = (2, 10, 0)
_ESCAPE_ENV = "EIGENFORGE_ALLOW_UNSAFE_TORCH"
_ADVISORIES = (
    "CVE-2025-32434 / GHSA-53q9-r3pm-6pq6 (fixed 2.6.0), "
    "CVE-2026-24747 / GHSA-63cw-57p8-fm3p (fixed 2.10.0)"
)


class UnsafeTorchError(RuntimeError):
    """Raised when the runtime torch cannot be trusted with weights_only loads."""


def parse_version(version: str) -> tuple[int, ...]:
    """Parse the PEP 440 public release portion, preserving prerelease information."""
    base = str(version).strip().split("+", 1)[0]
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?(?P<suffix>.*)$", base)
    if not match:
        return (0, 0, 0, 0)
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3)) if match.group(3) is not None else 0
    suffix = match.group("suffix")
    # A release candidate/dev/beta/alpha before the floor is not the patched
    # stable release. Keep a prerelease marker in the comparable tuple so
    # 2.10.0rc1 < 2.10.0, while PEP 440 local build metadata (+cu126, +ubuntu)
    # stripped above remains safe.
    prerelease = bool(re.match(r"^\.?(?:a|b|rc|dev)", suffix, re.IGNORECASE))
    return (major, minor, patch, -1 if prerelease else 0)


def is_safe_torch(version: str | None = None) -> bool:
    """Whether a stable runtime version is >= MIN_SAFE_TORCH."""
    candidate = torch.__version__ if version is None else version
    parsed = parse_version(candidate)
    floor = MIN_SAFE_TORCH + (0,)
    return parsed >= floor and parsed[-1] == 0


def require_safe_torch(*, when: str) -> None:
    """Fail loudly before loading checkpoints on a vulnerable torch.

    ``EIGENFORGE_ALLOW_UNSAFE_TORCH=1`` overrides with a loud warning, for
    developers who cannot upgrade yet; it never silences the default path.
    """
    if is_safe_torch():
        return
    if os.environ.get(_ESCAPE_ENV) == "1":
        message = (
            f"{when}: torch {torch.__version__} is below the safe floor "
            f"{'.'.join(map(str, MIN_SAFE_TORCH))}; torch.load(weights_only=True) "
            f"does not immunize this version against {_ADVISORIES}. "
            f"Escape hatch {_ESCAPE_ENV}=1 in use — "
            "upgrade: pip install -U torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        )
        # Dual channel: stderr for the operator watching this process, and the
        # loguru file sink so the fact that an RCE-floor escape was taken
        # survives in LOG_FILE even if stderr is discarded (an escape at
        # CVE-2025-32434 level must never be invisible after the fact).
        print(f"WARNING: {message}", file=sys.stderr)
        logger.warning(message)
        return
    raise UnsafeTorchError(
        f"{when}: torch {torch.__version__} < {'.'.join(map(str, MIN_SAFE_TORCH))} "
        "cannot load checkpoints safely: even weights_only=True is exploitable "
        f"on this version ({_ADVISORIES}). Upgrade with "
        "pip install -U torch --index-url "
        "https://download.pytorch.org/whl/cpu (torchvision/torchaudio must "
        "match the release matrix), or export "
        f"{_ESCAPE_ENV}=1 in the process environment (not .env) to proceed "
        "unsafely at your own risk."
    )


def enforce_safe_torch_at_startup(*, when: str) -> None:
    """CLI wrapper: turn the guard into an explicit exit instead of a traceback."""
    try:
        require_safe_torch(when=when)
    except UnsafeTorchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
