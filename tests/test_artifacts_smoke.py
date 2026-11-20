"""Slice 6 契约测试：模型产物追溯（ArtifactManifest）与 GPU 冒烟协议。

冒烟在 CPU 上也必须完整执行并如实报告 ``cuda_available=false``、
``quality_gate="not_evaluated"``——本地无 GPU 时不得伪造任何 GPU 指标。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------- ArtifactManifest


class TestArtifactManifest:
    def test_round_trip_records_checksum_and_fields(self, tmp_path):
        from eigenforge.artifacts import ArtifactManifest

        artifact = tmp_path / "ckpt.pt"
        artifact.write_bytes(b"\x89PK-fake-checkpoint-payload")
        manifest = ArtifactManifest.create(
            artifact_path=artifact,
            model_kind="eigengan",
            seed=42,
            step=1000,
            training_config={"lr": 0.0002, "ortho_weight": 1.0},
            dataset={"name": "celeba", "hash": "deadbeef"},
            metrics={"loss_g": 0.23},
        )
        out = tmp_path / "ckpt.manifest.json"
        manifest.to_json(out)
        loaded = json.loads(out.read_text(encoding="utf-8"))

        assert loaded["artifact_sha256"] == _sha256(artifact)
        assert loaded["model_kind"] == "eigengan"
        assert loaded["seed"] == 42
        assert loaded["step"] == 1000
        assert loaded["training_config"] == {"lr": 0.0002, "ortho_weight": 1.0}
        assert loaded["dataset"] == {"name": "celeba", "hash": "deadbeef"}
        assert loaded["metrics"] == {"loss_g": 0.23}
        assert loaded["created_utc"]
        # git 元数据：在 Git 仓库内应记录 40 位 sha 与布尔 dirty；在无 .git 的源码归档中允许为 None
        if loaded["git"] is not None:
            assert isinstance(loaded["git"], dict)
            assert isinstance(loaded["git"]["commit"], str) and len(loaded["git"]["commit"]) == 40
            assert isinstance(loaded["git"]["dirty"], bool)

    def test_missing_artifact_raises(self, tmp_path):
        from eigenforge.artifacts import ArtifactManifest

        with pytest.raises(FileNotFoundError):
            ArtifactManifest.create(
                artifact_path=tmp_path / "nope.pt",
                model_kind="diffusion",
                seed=0,
                step=1,
                training_config={},
            )

    def test_invalid_model_kind_raises(self, tmp_path):
        from eigenforge.artifacts import ArtifactManifest

        artifact = tmp_path / "ckpt.pt"
        artifact.write_bytes(b"x")
        with pytest.raises(ValueError):
            ArtifactManifest.create(
                artifact_path=artifact, model_kind="unknown-net", seed=0, step=1, training_config={}
            )

    def test_negative_step_raises(self, tmp_path):
        from eigenforge.artifacts import ArtifactManifest

        artifact = tmp_path / "ckpt.pt"
        artifact.write_bytes(b"x")
        with pytest.raises(ValueError):
            ArtifactManifest.create(
                artifact_path=artifact, model_kind="diffusion", seed=0, step=-1, training_config={}
            )


# ------------------------------------------------------- 冒烟协议


@pytest.fixture(scope="module")
def cpu_smoke_report():
    from eigenforge.smoke import run_smoke

    return run_smoke(seeds=(0,), steps=2, device=None)


class TestSmokeProtocol:
    def test_report_schema_and_honest_cpu_flags(self, cpu_smoke_report):
        import torch

        report = cpu_smoke_report
        assert report["report_schema_version"] == REPORT_SCHEMA_VERSION
        assert report["seeds"] == [0]
        assert report["steps_per_seed"] == 2
        # 质量门禁只在固定 GPU 环境评估；本地必须如实标注
        assert report["cuda_available"] is torch.cuda.is_available()
        if not torch.cuda.is_available():
            assert report["device"] == "cpu"
        assert report["quality_gate"] == "not_evaluated"
        if report["git"] is not None:
            assert isinstance(report["git"], dict)
        assert report["elapsed_total_seconds"] >= 0

    def test_models_executed_real_steps_without_nan(self, cpu_smoke_report):
        per_seed = cpu_smoke_report["per_seed"]
        assert len(per_seed) == 1
        entry = per_seed[0]
        assert entry["seed"] == 0
        for model_key in ("eigengan", "diffusion"):
            stats = entry[model_key]
            assert stats["steps_executed"] == 2
            assert stats["loss_finite"] is True
            assert stats["elapsed_seconds"] >= 0
            assert len(stats["sample_shape"]) == 4
            assert all(dim > 0 for dim in stats["sample_shape"])

    def test_multi_seed_aggregates_mean_std(self):
        from eigenforge.smoke import run_smoke

        report = run_smoke(seeds=(1, 2), steps=1, device=None)
        assert report["seeds"] == [1, 2]
        assert len(report["per_seed"]) == 2
        assert {item["seed"] for item in report["per_seed"]} == {1, 2}
        agg = report["aggregate"]
        for metric in ("eigengan_loss_g", "diffusion_loss"):
            stats = agg[metric]
            assert set(stats) == {"mean", "std", "min", "max"}
            assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_aggregate_seed_metrics_pure_function(self):
        from eigenforge.smoke import aggregate_seed_metrics

        per_seed = [
            {"seed": 0, "eigengan": {"loss_g": 1.0}, "diffusion": {"loss": 2.0}},
            {"seed": 1, "eigengan": {"loss_g": 3.0}, "diffusion": {"loss": 6.0}},
        ]
        agg = aggregate_seed_metrics(per_seed)
        assert agg["eigengan_loss_g"] == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0}
        assert agg["diffusion_loss"]["mean"] == pytest.approx(4.0)


# ------------------------------------------------------- CLI


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "scripts" / "gpu_smoke.py"), *args],
        cwd=PACKAGE_ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_cli_writes_report_and_exits_zero(tmp_path):
    out = tmp_path / "gpu_smoke.json"
    result = _run_cli("--seeds", "0", "--steps", "1", "--output", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert report["quality_gate"] == "not_evaluated"
    assert report["seeds"] == [0]
    assert "SMOKE" in result.stdout


def test_smoke_diffusion_uses_production_data_range():
    """The diffusion half must train/sample on [-1, 1], like production.

    train_diffusion.py normalises with mean/std 0.5 and the scheduler clamps
    predicted x0 to [-1, 1]; feeding [0, 1] data here meant the clamp never
    engaged and the half did not exercise the production range at all.
    """
    import torch

    from eigenforge.models.diffusion_scheduler import DiffusionScheduler
    from eigenforge.smoke import _smoke_diffusion

    captured: dict[str, float] = {}
    original = DiffusionScheduler.q_sample_with_noise

    def spy(self, x_start, t, *args, **kwargs):
        captured["min"] = float(x_start.min())
        captured["max"] = float(x_start.max())
        return original(self, x_start, t, *args, **kwargs)

    DiffusionScheduler.q_sample_with_noise = spy
    try:
        _smoke_diffusion(torch.device("cpu"), 1)
    finally:
        DiffusionScheduler.q_sample_with_noise = original

    assert captured["min"] < 0.0, "diffusion half still feeds [0,1] data"
    assert captured["max"] <= 1.0
