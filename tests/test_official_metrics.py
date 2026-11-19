from __future__ import annotations

import importlib.util

import pytest
import torch

from eigenforge.evaluation.official_metrics import (
    SeedEvidence,
    compute_clean_fid,
    compute_torch_fidelity,
    official_metrics_status,
    three_seed_evidence,
)


def test_missing_backend_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert official_metrics_status("cleanfid")["status"] == "not-configured"
    with pytest.raises(RuntimeError, match="clean-fid"):
        compute_clean_fid("real", "fake")
    with pytest.raises(RuntimeError, match="torch-fidelity"):
        compute_torch_fidelity([], [])


def _rows(**overrides):
    base = {"dataset_version": "ds-v1", "device": "cpu", "torch_version": "2.10"}
    base.update(overrides)
    return [
        SeedEvidence(
            i,
            f"sha-{i}",
            base["dataset_version"],
            base["device"],
            base["torch_version"],
            {"fid": float(i), "is": 1 + i},
        )
        for i in (0, 1, 2)
    ]


def test_three_seed_evidence_is_pure_and_aggregates_scores() -> None:
    report = three_seed_evidence(_rows())
    assert report["seed_count"] == 3
    assert report["dataset_version"] == "ds-v1"
    assert report["scores"]["fid"]["mean"] == 1.0
    assert report["scores"]["fid"]["std"] == pytest.approx(2**0.5 / 3**0.5)
    assert report["production_gate"] is False
    assert [row["seed"] for row in report["runs"]] == [0, 1, 2]


def test_three_seed_reports_per_seed_checkpoints_instead_of_the_first_one() -> None:
    # One checkpoint per seed is the normal case: a single "checkpoint_sha" would
    # silently attribute the aggregate to seed 0's checkpoint only.
    report = three_seed_evidence(_rows())
    assert "checkpoint_sha" not in report
    assert report["checkpoint_shas"] == ["sha-0", "sha-1", "sha-2"]


def test_three_seed_collapses_to_single_checkpoint_only_when_identical() -> None:
    rows = [
        {"seed": i, "checkpoint_sha": "same", "dataset_version": "d", "device": "cpu",
         "torch_version": "t", "scores": {"fid": float(i)}}
        for i in (0, 1, 2)
    ]
    report = three_seed_evidence(rows)
    assert report["checkpoint_sha"] == "same"
    assert "checkpoint_shas" not in report


@pytest.mark.parametrize(
    "field,bad",
    [("dataset_version", "ds-v2"), ("device", "cuda"), ("torch_version", "2.9")],
)
def test_three_seed_rejects_incomparable_provenance(field, bad) -> None:
    rows = _rows()
    rows[1] = SeedEvidence(
        1, "sha-1",
        bad if field == "dataset_version" else "ds-v1",
        bad if field == "device" else "cpu",
        bad if field == "torch_version" else "2.10",
        {"fid": 1.0, "is": 2.0},
    )
    with pytest.raises(ValueError, match=f"disagree on {field}"):
        three_seed_evidence(rows)
    # An explicit, caller-owned override is still honored.
    assert three_seed_evidence(rows, **{field: "verified-comparable"})[field] == "verified-comparable"


def test_three_seed_rejects_metrics_missing_from_some_seeds() -> None:
    rows = _rows()
    rows[2] = SeedEvidence(2, "sha-2", "ds-v1", "cpu", "2.10", {"fid": 2.0})  # no "is"
    with pytest.raises(ValueError, match="different metric names"):
        three_seed_evidence(rows)


def test_three_seed_requires_distinct_three_records() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        three_seed_evidence([])
    rows = [
        {"seed": 1, "checkpoint_sha": "x", "dataset_version": "d", "device": "cpu", "torch_version": "t", "scores": {"fid": 1}},
    ] * 3
    with pytest.raises(ValueError, match="distinct"):
        three_seed_evidence(rows)


def test_safetensors_roundtrip_preserves_tensor_weights(tmp_path):
    # safetensors is an optional extra (pip install -e ".[safetensors]"); a clean
    # ".[dev]" environment must skip, not fail.
    pytest.importorskip("safetensors")
    from eigenforge.safetensors_io import export_state_dict, import_state_dict, metadata

    path = tmp_path / "generator.safetensors"
    export_state_dict(
        {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        path,
        metadata={"checkpoint_sha": "abc", "model_kind": "eigengan"},
    )
    loaded = import_state_dict(path)
    assert torch.equal(loaded["weight"], torch.arange(4, dtype=torch.float32).reshape(2, 2))
    assert metadata(path)["checkpoint_sha"] == "abc"


def test_safetensors_helpers_import_without_the_optional_dependency() -> None:
    # The module must be importable (and fail with a pointed message when used)
    # even where safetensors is not installed.
    from eigenforge import safetensors_io

    assert callable(safetensors_io.export_state_dict)
