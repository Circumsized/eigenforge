"""Optional official metric adapters and an isolated multi-seed evidence protocol.

The adapters are deliberately not imported by :mod:`evaluator` and never feed a
training or production quality gate.  The dependency-free evaluator therefore
continues to expose its proxy metrics unchanged.
"""
from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

PROTOCOL_VERSION = "eigenforge_official_metrics_v1"
_SUPPORTED = {"torch-fidelity", "cleanfid"}


@dataclass(frozen=True)
class BackendStatus:
    """Availability information that is safe to display in a report."""

    backend: str
    status: str
    error: str | None = None


def backend_status(backend: str) -> BackendStatus:
    """Return ``ok`` or ``not-configured`` without importing heavy backends."""
    name = backend.strip().lower()
    if name not in _SUPPORTED:
        raise ValueError(f"backend must be one of {sorted(_SUPPORTED)}")
    module = "torch_fidelity" if name == "torch-fidelity" else "cleanfid"
    if importlib.util.find_spec(module) is None:
        return BackendStatus(name, "not-configured", f"optional dependency not installed: {module}")
    return BackendStatus(name, "ok")


def _missing_dependency(backend: str, module: str) -> RuntimeError:
    return RuntimeError(
        f"official metric backend '{backend}' is not configured; install optional dependency '{module}'"
    )


def compute_torch_fidelity(
    real: Any,
    fake: Any,
    *,
    device: str = "cpu",
    kid: bool = True,
    fid: bool = True,
    isc: bool = True,
) -> dict[str, float]:
    """Compute torch-fidelity metrics for two tensor/image-folder inputs.

    This function is an explicit opt-in boundary.  It raises a clear
    ``RuntimeError`` when the optional package is absent and returns only the
    backend's metric scores; callers decide how to report them.
    """
    if importlib.util.find_spec("torch_fidelity") is None:
        raise _missing_dependency("torch-fidelity", "torch-fidelity")
    from torch_fidelity import calculate_metrics

    result = calculate_metrics(
        input1=real,
        input2=fake,
        cuda=str(device).startswith("cuda"),
        kid=bool(kid),
        fid=bool(fid),
        isc=bool(isc),
        verbose=False,
    )
    return {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float))}


def compute_clean_fid(real_dir: str, fake_dir: str, *, device: str = "cpu") -> float:
    """Compute Clean-FID, retaining the optional dependency boundary."""
    if importlib.util.find_spec("cleanfid") is None:
        raise _missing_dependency("cleanfid", "clean-fid")
    from cleanfid import fid

    return float(fid.compute_fid(real_dir, fake_dir, device=device))


def official_metrics_status(backend: str) -> dict[str, Any]:
    """JSON-friendly status envelope; missing dependencies are not failures."""
    return asdict(backend_status(backend))


@dataclass(frozen=True)
class SeedEvidence:
    seed: int
    checkpoint_sha: str
    dataset_version: str
    device: str
    torch_version: str
    scores: Mapping[str, float]


def three_seed_evidence(
    records: Sequence[SeedEvidence | Mapping[str, Any]],
    *,
    checkpoint_sha: str | None = None,
    dataset_version: str | None = None,
    device: str | None = None,
    torch_version: str | None = None,
) -> dict[str, Any]:
    """Build deterministic evidence for exactly three seed runs.

    This is a pure aggregation function: it performs no I/O, imports no metric
    backend, and has no connection to a production quality gate.  Each record
    retains its provenance, while numeric scores receive population mean/std.
    """
    if len(records) != 3:
        raise ValueError("three_seed_evidence requires exactly three seed records")
    normalized: list[dict[str, Any]] = []
    for item in records:
        row = asdict(item) if isinstance(item, SeedEvidence) else dict(item)
        required = ("seed", "checkpoint_sha", "dataset_version", "device", "torch_version", "scores")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"seed record missing fields: {', '.join(missing)}")
        if not isinstance(row["scores"], Mapping):
            raise TypeError("scores must be a mapping")
        scores = {str(key): float(value) for key, value in row["scores"].items()}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError("scores must be finite")
        row["seed"] = int(row["seed"])
        row["scores"] = scores
        normalized.append(row)
    if len({row["seed"] for row in normalized}) != 3:
        raise ValueError("seed records must contain three distinct seeds")
    metric_sets = [frozenset(row["scores"]) for row in normalized]
    if len(set(metric_sets)) != 1:
        raise ValueError(
            "seed records report different metric names "
            f"({[sorted(names) for names in metric_sets]}); a mean/std over a metric "
            "that only some seeds measured would be silently misleading"
        )
    names = sorted(metric_sets[0])
    aggregates: dict[str, dict[str, float]] = {}
    for name in names:
        values = [row["scores"][name] for row in normalized]
        aggregates[name] = {"mean": mean(values), "std": pstdev(values)}
    # Provenance that must be shared for a 3-seed aggregate to be meaningful:
    # an explicit argument overrides the check, otherwise disagreement is an error
    # instead of "whatever the first record said".
    provenance: dict[str, Any] = {}
    for key, override in (
        ("dataset_version", dataset_version),
        ("device", device),
        ("torch_version", torch_version),
    ):
        seen = {str(row[key]) for row in normalized}
        if override is not None:
            provenance[key] = override
        elif len(seen) == 1:
            provenance[key] = normalized[0][key]
        else:
            raise ValueError(
                f"seed records disagree on {key}: {sorted(seen)}; the runs are not comparable "
                f"(pass {key}=... explicitly only after verifying they are)"
            )
    # One checkpoint per seed is the normal case, so a single "checkpoint_sha"
    # would be a lie; report the per-seed list unless they are genuinely identical.
    checkpoint_shas = [row["checkpoint_sha"] for row in sorted(normalized, key=lambda r: r["seed"])]
    if checkpoint_sha is not None:
        provenance["checkpoint_sha"] = checkpoint_sha
    elif len(set(checkpoint_shas)) == 1:
        provenance["checkpoint_sha"] = checkpoint_shas[0]
    else:
        provenance["checkpoint_shas"] = checkpoint_shas
    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed_count": 3,
        **provenance,
        "runs": normalized,
        "scores": aggregates,
        "production_gate": False,
    }


# Descriptive aliases for callers that prefer verb-first names.
collect_three_seed_evidence = three_seed_evidence
get_backend_status = official_metrics_status
