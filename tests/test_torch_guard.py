"""Wave-8 guard: torch.load safety (weights_only + CVE-2025-32434 / CVE-2026-24747 floor).

Two layers:

* Every ``torch.load`` in product source must pass ``weights_only=True``
  (AGENTS.md §3-3). AST-scanned dynamically — new call sites are covered
  automatically, including the ones inside ``checkpoints.py``.
* ``weights_only=True`` is only a boundary on a patched torch. GHSA-53q9-r3pm-6pq6
  / CVE-2025-32434 (CVSS v4 9.3) made it RCE-exploitable on <= 2.5.1 (fixed
  2.6.0), and GHSA-63cw-57p8-fm3p / CVE-2026-24747 (high) made the unpickler
  memory-corruptible on < 2.10.0 (fixed 2.10.0). The runtime torch must meet
  the 2.10.0 floor; the floor assertion is enforced on patched runners (CI
  installs the latest CPU torch) and visibly *skipped* with the CVEs in the
  reason on older local installs — honest, never a silent pass.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from eigenforge.common.torch_guard import (
    MIN_SAFE_TORCH,
    UnsafeTorchError,
    is_safe_torch,
    parse_version,
    require_safe_torch,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class _UnpicklableSentinel:
    """Module-level so pickle can reference it by qualified name."""


# ------------------------------------------------------- version parsing


class TestVersionMatrix:
    @pytest.mark.parametrize(
        "bad",
        [
            "1.13.1", "2.0.0", "2.5.0", "2.5.1", "2.5.1+cu121", "2.5.1.dev20250101", "1.0",
            # CVE-2025-32434 fixed these, CVE-2026-24747 (< 2.10.0) still covers them.
            "2.6.0", "2.6.0+cu124", "2.7.1", "2.8.0", "2.9.0", "2.9.1", "2.9.1+cu126",
        ],
    )
    def test_vulnerable_versions_rejected(self, bad):
        assert is_safe_torch(bad) is False

    @pytest.mark.parametrize(
        "good",
        [
            "2.10.0",
            "2.10.0+cu126",
            "2.10.0+cuda126",
            "2.10.0+ubuntu",
            "2.10.0+git1a2b3c",
            "2.10",
            "2.10.1",
            "2.14.1",
            "2.14.1+cpu",
            "3.0.0",
        ],
    )
    def test_patched_versions_accepted(self, good):
        assert is_safe_torch(good) is True

    @pytest.mark.parametrize(
        "prerelease",
        ["2.10.0.dev0", "2.10.0rc1", "2.10.0a1", "2.10.0b1", "2.9.99"],
    )
    def test_prerelease_versions_are_not_the_patched_stable_floor(self, prerelease):
        assert is_safe_torch(prerelease) is False

    def test_parse_version_ignores_build_suffixes(self):
        assert parse_version("2.10.0+cu126")[0:3] == (2, 10, 0)
        assert parse_version("2.5.1.dev20250101")[0:3] == (2, 5, 1)

    def test_min_safe_torch_matches_cve_fix_line(self):
        # 2.10.0 is where GHSA-63cw-57p8-fm3p (CVE-2026-24747) was fixed; it
        # subsumes the 2.6.0 line of GHSA-53q9-r3pm-6pq6 (CVE-2025-32434).
        assert MIN_SAFE_TORCH == (2, 10, 0)
        assert MIN_SAFE_TORCH >= (2, 6, 0)


# ------------------------------------------------------- runtime guard


class TestRequireSafeTorch:
    def test_vulnerable_runtime_raises_with_cve_reference(self, monkeypatch):
        monkeypatch.delenv("EIGENFORGE_ALLOW_UNSAFE_TORCH", raising=False)
        monkeypatch.setattr(torch, "__version__", "2.5.1+cu121")
        with pytest.raises(UnsafeTorchError) as excinfo:
            require_safe_torch(when="unit-test")
        assert "CVE-2025-32434" in str(excinfo.value)
        assert "CVE-2026-24747" in str(excinfo.value)

    def test_runtime_between_old_and_new_fix_is_still_rejected(self, monkeypatch):
        # 2.6.0 fixed CVE-2025-32434 only; CVE-2026-24747 still applies.
        monkeypatch.delenv("EIGENFORGE_ALLOW_UNSAFE_TORCH", raising=False)
        monkeypatch.setattr(torch, "__version__", "2.9.1")
        with pytest.raises(UnsafeTorchError) as excinfo:
            require_safe_torch(when="unit-test")
        assert "CVE-2026-24747" in str(excinfo.value)

    def test_escape_hatch_overrides_with_visible_warning(self, monkeypatch, capsys):
        monkeypatch.setattr(torch, "__version__", "2.5.1")
        monkeypatch.setenv("EIGENFORGE_ALLOW_UNSAFE_TORCH", "1")
        require_safe_torch(when="unit-test")  # must not raise
        stderr = capsys.readouterr().err
        assert "CVE-2025-32434" in stderr and "CVE-2026-24747" in stderr

    def test_escape_hatch_must_be_exactly_one(self, monkeypatch):
        monkeypatch.setattr(torch, "__version__", "2.5.1")
        monkeypatch.setenv("EIGENFORGE_ALLOW_UNSAFE_TORCH", "0")
        with pytest.raises(UnsafeTorchError):
            require_safe_torch(when="unit-test")

    def test_patched_runtime_passes_silently(self, monkeypatch):
        # On CI (latest CPU torch) this is the live path: no raise, no warning.
        monkeypatch.setattr(torch, "__version__", "2.10.0")
        require_safe_torch(when="unit-test")


class TestStartupWrapper:
    @pytest.mark.skipif(
        is_safe_torch(),
        reason="local torch already meets the 2.10.0 floor; the guard does not fire here",
    )
    def test_cli_fails_closed_with_exit_two(self):
        # Subprocess with the real torch: the wrapper must exit 2, not traceback.
        child_env = {
            k: v for k, v in os.environ.items() if k != "EIGENFORGE_ALLOW_UNSAFE_TORCH"
        }
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from eigenforge.common.torch_guard import enforce_safe_torch_at_startup;"
                "enforce_safe_torch_at_startup(when='cli-test')",
            ],
            cwd=PACKAGE_ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 2
        assert "CVE-2025-32434" in proc.stderr

    def test_runtime_torch_meets_weights_only_hardening_floor(self):
        if not is_safe_torch():
            pytest.skip(
                f"local torch {torch.__version__} < 2.10.0: CVE-2025-32434 "
                "(weights_only=True RCE, fixed 2.6.0) and CVE-2026-24747 "
                "(weights_only unpickler memory corruption, fixed 2.10.0) — "
                "install a patched torch or use the CPU index to enforce this assertion"
            )
        assert is_safe_torch()


# ------------------------------------------------------- static torch.load guard


def _product_sources() -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    excluded = {
        "tests",
        "__pycache__",
        "outputs",
        "logs",
        ".zcode",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".git",
    }
    files = [
        path
        for path in project_root.rglob("*.py")
        if not (excluded & set(path.relative_to(project_root).parts))
        and path.name != "conftest.py"
    ]
    assert files, "no source files discovered"
    return files


def _torch_load_violations(tree: ast.AST) -> list[str]:
    """Every ``torch.load``-equivalent call must pass ``weights_only=True``.

    Both import forms are tracked so an alias cannot sidestep the guard:
    ``import torch`` / ``import torch as t`` and ``from torch import load``.
    The keyword's *value* is checked, not just its presence: passing
    ``weights_only=False`` (or ``None``) is exactly the CVE-2025-32434
    degradation the flag exists to prevent. A ``**kwargs`` expansion is not
    statically inspectable and is therefore flagged (fail-closed direction).
    """
    torch_aliases = {"torch"}
    load_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            torch_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "torch"
            )
        if isinstance(node, ast.ImportFrom) and node.module == "torch":
            load_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "load"
            )
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            is_torch_load = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in torch_aliases
            ) or (isinstance(node.func, ast.Name) and node.func.id in load_aliases)
            if not is_torch_load:
                continue
            keyword = next((kw for kw in node.keywords if kw.arg == "weights_only"), None)
            if keyword is None:
                offenders.append(f"line {node.lineno}: torch.load missing weights_only=True")
            elif not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            ):
                offenders.append(
                    f"line {node.lineno}: torch.load weights_only must be literal True"
                )
    return offenders


class TestTorchLoadGuard:
    def test_every_product_torch_load_is_weights_only(self):
        offenders: list[str] = []
        sites_by_file: dict[str, int] = {}
        for path in _product_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(PACKAGE_ROOT).as_posix()

            # count call sites with the same predicate that flags violations
            class _Counter(ast.NodeVisitor):
                def visit_Call(self, node: ast.Call) -> None:
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "load"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in {"torch"}
                    ) or (
                        isinstance(node.func, ast.Name)
                        and node.func.id in {"load", "L"}
                    ):
                        sites_by_file[relative] = sites_by_file.get(relative, 0) + 1
                    self.generic_visit(node)

            _Counter().visit(tree)
            offenders.extend(f"{relative}: {hit}" for hit in _torch_load_violations(tree))
        # Every checkpoint load funnels through checkpoints.load_checkpoint (guard
        # + single-read sha256 verification + schema checks). A second call site
        # elsewhere would bypass that verification, so it is a failure on its own.
        # The floor also guards against the scan silently covering nothing.
        assert sites_by_file, "scan coverage dropped unexpectedly: no torch.load site found"
        assert set(sites_by_file) == {"src/eigenforge/checkpoints.py"}, (
            "torch.load must only be called from checkpoints.load_checkpoint "
            f"(route other loads through it): {sites_by_file}"
        )
        assert not offenders, "torch.load without weights_only=True:\n" + "\n".join(offenders)

    @pytest.mark.parametrize(
        "source",
        [
            "import torch\ntorch.load('a.pt', weights_only=True)\n",
            "import torch as t\nt.load('a.pt', weights_only=True)\n",
            "from torch import load\nload('a.pt', weights_only=True)\n",
            "from torch import load as L\nL('a.pt', weights_only=True)\n",
        ],
    )
    def test_alias_forms_pass_predicate(self, source):
        assert _torch_load_violations(ast.parse(source)) == []

    @pytest.mark.parametrize(
        "source",
        [
            "import torch\ntorch.load('a.pt')\n",
            "import torch as t\nt.load('a.pt', map_location='cpu')\n",
            "from torch import load\nload('a.pt')\n",
            # The CVE degradation form: the kwarg is present but disables it.
            "import torch\ntorch.load('a.pt', weights_only=False)\n",
            "import torch\ntorch.load('a.pt', weights_only=None)\n",
            # Not statically inspectable: fail closed rather than assume.
            "import torch\ntorch.load('a.pt', **opts)\n",
        ],
    )
    def test_unprotected_forms_are_flagged(self, source):
        assert _torch_load_violations(ast.parse(source))

    def test_non_torch_loads_are_not_mistaken_for_torch_load(self):
        tree = ast.parse("import json\njson.load(open('x'))\nloader.load('y')\nnumpy.load('z')\n")
        assert _torch_load_violations(tree) == []


# ------------------------------------------------------- behavior of weights_only itself


class TestWeightsOnlyBehavior:
    def test_weights_only_rejects_non_allowlisted_object(self, tmp_path):
        import io
        import pickle

        blob = io.BytesIO(pickle.dumps(_UnpicklableSentinel()))
        with pytest.raises(Exception):
            torch.load(blob, map_location="cpu", weights_only=True)

    def test_weights_only_roundtrips_plain_state_dicts(self, tmp_path):
        path = tmp_path / "sd.pt"
        torch.save({"w": torch.ones(3)}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(loaded["w"], torch.ones(3))
