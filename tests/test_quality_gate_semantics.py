"""Fail-closed semantics of the quality gate and the GPU smoke CLI.

The gate must never turn "could not measure" or "measured garbage" into a pass:

* a NaN/inf metric is a breach (``mean > limit`` is False for NaN, so the
  comparison is written as ``not mean <= limit``);
* a NaN/inf threshold is rejected (``mean > nan`` waves everything through);
* staying inside thresholds on a CPU-only tiny synthetic protocol certifies
  nothing, so it is ``not_evaluated`` — only CUDA can yield ``pass``;
* ``gpu_smoke --require-cuda`` refuses to run on a machine without CUDA.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_p1_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def gate(monkeypatch):
    module = _load_script("quality_gate")
    healthy = {"eigengan_loss_g": 0.5, "eigengan_ortho_error": 1e-6, "diffusion_loss": 1.0}
    monkeypatch.setattr(module, "_run_seed", lambda seed, steps, device: {"seed": seed, **healthy})
    return module


def _evaluate(module, **thresholds):
    return module.run_gate((0,), 1, "evaluate", thresholds or {"max_ortho_error": 1.0})


def test_cpu_run_inside_thresholds_is_not_evaluated_never_pass(gate, monkeypatch):
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: False)
    report = _evaluate(gate)
    assert report["breaches"] == []
    assert report["quality_gate"] == "not_evaluated"
    assert "CUDA" in report["not_evaluated_reason"]


def test_cuda_run_inside_thresholds_passes(gate, monkeypatch):
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    assert _evaluate(gate)["quality_gate"] == "pass"


def test_breach_fails_on_any_device(gate, monkeypatch):
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: False)
    report = _evaluate(gate, max_diffusion_loss=0.1)
    assert report["quality_gate"] == "fail"
    assert any("diffusion_loss" in breach for breach in report["breaches"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_is_a_breach_not_a_pass(gate, monkeypatch, bad):
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gate, "_run_seed",
        lambda seed, steps, device: {
            "seed": seed, "eigengan_loss_g": 0.5, "eigengan_ortho_error": 1e-6, "diffusion_loss": bad,
        },
    )
    report = _evaluate(gate, max_diffusion_loss=1e9)
    assert report["quality_gate"] == "fail"
    assert any("diffusion_loss" in breach for breach in report["breaches"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_threshold_is_rejected_by_run_gate(gate, bad):
    with pytest.raises(ValueError, match="finite"):
        _evaluate(gate, max_ortho_error=bad)


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_non_finite_threshold_exits_2_from_the_cli(gate, bad, capsys):
    code = gate.main(["--mode", "evaluate", "--seeds", "0", "--train-steps", "1", "--max-ortho-error", bad])
    assert code == 2
    assert "finite" in capsys.readouterr().err


def test_evaluate_cli_exits_2_when_it_cannot_certify(gate, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: False)
    out = tmp_path / "gate.json"
    code = gate.main([
        "--mode", "evaluate", "--seeds", "0", "--train-steps", "1",
        "--max-ortho-error", "1", "--output", str(out),
    ])
    assert code == 2, "a CI step must not read 'refused to judge' as success"
    assert "not certified" in capsys.readouterr().err


def test_protocol_mode_is_unchanged_and_exits_zero(gate, tmp_path):
    out = tmp_path / "gate.json"
    assert gate.main(["--mode", "protocol", "--seeds", "0", "--train-steps", "1", "--output", str(out)]) == 0


def test_gpu_smoke_require_cuda_refuses_without_a_device(monkeypatch, capsys):
    smoke = _load_script("gpu_smoke")
    monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: False)

    def _must_not_run(**kwargs):
        raise AssertionError("smoke ran although --require-cuda could not be satisfied")

    monkeypatch.setattr(smoke, "run_smoke", _must_not_run)
    assert smoke.main(["--require-cuda", "--seeds", "0"]) == 2
    assert "--require-cuda" in capsys.readouterr().err


def test_gpu_smoke_on_cpu_says_it_is_not_a_gpu_verification(monkeypatch, tmp_path, capsys):
    smoke = _load_script("gpu_smoke")
    monkeypatch.setattr(
        smoke, "run_smoke",
        lambda **kwargs: {
            "device": "cpu", "cuda_available": False, "seeds": [0], "steps_per_seed": 1,
            "quality_gate": "not_evaluated", "aggregate": {},
        },
    )
    assert smoke.main(["--seeds", "0", "--steps", "1", "--output", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "SMOKE OK" in out and "NOT a GPU/AMP verification" in out
