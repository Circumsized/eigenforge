"""Bridge ``eigenforge.scripts.*`` to the top-level ``scripts/`` CLI directory."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[3] / "scripts")]
