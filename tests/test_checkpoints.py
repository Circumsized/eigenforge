"""Wave-8 contract: durable checkpoint save/load with manifest verification.

Locks in the properties the two training entry points now share (they used to
each hand-roll them, and both hand-rolls were weaker): atomic+durable writes,
sibling traceability manifest, sha256 verification on resume, strict epoch
schema, architecture anti-drift, and NaN/Inf loss circuit-breaking.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

from eigenforge.checkpoints import (
    CheckpointIntegrityError,
    CheckpointSchemaError,
    NonFiniteLossError,
    check_finite_losses,
    check_finite_parameters,
    load_checkpoint,
    save_checkpoint,
    select_generator_state,
)
from eigenforge.common.fsutil import atomic_write_json, durable_replace_bytes
from eigenforge.models.diffusion_scheduler import DiffusionScheduler
from eigenforge.models.eigengan import EigenDiscriminator, EigenGenerator


@pytest.fixture(autouse=True)
def _allow_legacy_local_torch_for_checkpoint_unit_tests(monkeypatch):
    """The reference interpreter is torch 2.5.1; guard behavior is tested separately."""
    monkeypatch.setenv("EIGENFORGE_ALLOW_UNSAFE_TORCH", "1")


def _tiny_payload(epoch: int = 3) -> dict:
    generator = EigenGenerator(img_size=8, latent_dim=4, n_basis=2, base_channels=8, max_channels=16)
    discriminator = EigenDiscriminator(img_size=8, base_channels=16)
    g_optim = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    return {
        "generator": generator.state_dict(),
        "ema_generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optim": g_optim.state_dict(),
        "d_optim": d_optim.state_dict(),
        "epoch": epoch,
    }


_CONFIG = {"z_dim": 4, "image_size": 8, "n_basis": 2}
_REQUIRED = ("generator", "discriminator", "g_optim", "d_optim")


def _save(tmp_path: Path, payload: dict | None = None, epoch: int = 3):
    return save_checkpoint(
        tmp_path / "latest.pt",
        _tiny_payload() if payload is None else payload,
        model_kind="eigengan",
        seed=42,
        step=600,
        training_config=dict(_CONFIG),
        dataset={"name": "celeba", "num_images": 1},
    )


# ------------------------------------------------------- save/load round trip


class TestSaveLoadContract:
    def test_save_writes_checkpoint_and_manifest_atomically(self, tmp_path):
        manifest = _save(tmp_path)
        artifact = tmp_path / "latest.pt"
        sidecar = tmp_path / "latest.pt.manifest.json"
        assert artifact.is_file() and sidecar.is_file()
        import hashlib
        import json

        assert manifest.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert loaded["artifact_sha256"] == manifest.artifact_sha256
        assert loaded["seed"] == 42 and loaded["step"] == 600
        assert loaded["training_config"] == _CONFIG

    def test_load_verifies_and_returns_structured_state(self, tmp_path):
        _save(tmp_path)
        state = load_checkpoint(
            tmp_path / "latest.pt",
            required_keys=_REQUIRED,
            expected_config=dict(_CONFIG),
        )
        assert state.epoch == 3
        assert set(state.payload) == set(_REQUIRED) | {"ema_generator", "epoch"}
        assert state.manifest is not None and state.manifest["seed"] == 42

    def test_tampered_checkpoint_fails_integrity_closed(self, tmp_path):
        artifact = tmp_path / "latest.pt"
        _save(tmp_path)
        data = bytearray(artifact.read_bytes())
        data[len(data) // 2] ^= 0xFF
        artifact.write_bytes(bytes(data))
        with pytest.raises(CheckpointIntegrityError) as excinfo:
            load_checkpoint(artifact, required_keys=_REQUIRED)
        assert "sha256" in str(excinfo.value)

    def test_corrupt_manifest_fails_closed(self, tmp_path):
        _save(tmp_path)
        (tmp_path / "latest.pt.manifest.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(CheckpointIntegrityError):
            load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED)

    def test_legacy_checkpoint_without_manifest_loads_unverified(self, tmp_path):
        artifact = tmp_path / "legacy.pt"
        durable_replace_bytes(artifact, lambda handle: torch.save(_tiny_payload(epoch=7), handle))
        state = load_checkpoint(artifact, required_keys=_REQUIRED)
        assert state.manifest is None
        assert state.epoch == 7

    def test_missing_required_key_raises_schema_error(self, tmp_path):
        artifact = tmp_path / "latest.pt"
        durable_replace_bytes(
            artifact,
            lambda handle: torch.save({"generator": {}, "epoch": 1}, handle),
        )
        with pytest.raises(CheckpointSchemaError) as excinfo:
            load_checkpoint(artifact, required_keys=_REQUIRED)
        assert "discriminator" in str(excinfo.value) and "g_optim" in str(excinfo.value)

    def test_missing_epoch_raises_instead_of_restarting_from_zero(self, tmp_path):
        artifact = tmp_path / "latest.pt"
        durable_replace_bytes(
            artifact,
            lambda handle: torch.save({"generator": {}, "discriminator": {}, "g_optim": {}, "d_optim": {}}, handle),
        )
        with pytest.raises(CheckpointSchemaError) as excinfo:
            load_checkpoint(artifact, required_keys=_REQUIRED)
        assert "epoch" in str(excinfo.value)

    def test_architecture_drift_is_rejected(self, tmp_path):
        _save(tmp_path)
        with pytest.raises(CheckpointSchemaError) as excinfo:
            load_checkpoint(
                tmp_path / "latest.pt",
                required_keys=_REQUIRED,
                expected_config={**_CONFIG, "z_dim": 999},
            )
        assert "999" in str(excinfo.value)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nope.pt")


# ------------------------------------------------------- durability


class TestDurableWrites:
    def test_failed_write_preserves_previous_checkpoint(self, tmp_path):
        artifact = tmp_path / "latest.pt"
        durable_replace_bytes(artifact, lambda handle: handle.write(b"good-v1"))

        def _boom(handle):
            handle.write(b"half")
            raise RuntimeError("disk full mid-save")

        with pytest.raises(RuntimeError):
            durable_replace_bytes(artifact, _boom)
        assert artifact.read_bytes() == b"good-v1"
        residue = [p.name for p in tmp_path.iterdir() if p.name != artifact.name]
        assert residue == [], f"temp files left behind: {residue}"

    def test_successful_write_replaces_atomically(self, tmp_path):
        artifact = tmp_path / "blob.bin"
        durable_replace_bytes(artifact, lambda handle: handle.write(b"first"))
        durable_replace_bytes(artifact, lambda handle: handle.write(b"second"))
        assert artifact.read_bytes() == b"second"

    def test_atomic_write_json_roundtrip(self, tmp_path):
        out = tmp_path / "nested" / "report.json"
        atomic_write_json(out, {"b": 1, "a": [1, 2]})
        import json

        assert json.loads(out.read_text(encoding="utf-8")) == {"b": 1, "a": [1, 2]}


# ------------------------------------------------------- NaN/Inf circuit breaker


class TestNonFiniteGuard:
    def test_finite_losses_pass(self):
        check_finite_losses({"d_loss": torch.tensor(1.5), "g_loss": torch.tensor(0.0)}, epoch=0, step=0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_loss_raises_with_context(self, bad):
        with pytest.raises(NonFiniteLossError) as excinfo:
            check_finite_losses({"d_loss": torch.tensor(bad)}, epoch=2, step=5)
        message = str(excinfo.value)
        assert "epoch=2" in message and "step=5" in message and "d_loss" in message

    def test_only_offending_key_is_reported(self):
        with pytest.raises(NonFiniteLossError) as excinfo:
            check_finite_losses(
                {"ok": torch.tensor(1.0), "bad": torch.tensor(float("nan"))}, epoch=0, step=0
            )
        assert "bad" in str(excinfo.value) and "'ok'" not in str(excinfo.value)


# ------------------------------------------------------- entry-point contracts


class TestEntryPointContract:
    def test_eigengan_parse_args_defaults(self):
        from eigenforge import train_eigengan

        with mock.patch.object(sys, "argv", ["train_eigengan"]):
            args = train_eigengan.parse_args()
        assert args.deterministic is False
        assert args.z_dim == 128
        assert args.output_dir == "./experiments/eigengan"

    def test_diffusion_parse_args_defaults(self):
        from eigenforge import train_diffusion

        with mock.patch.object(sys, "argv", ["train_diffusion"]):
            args = train_diffusion.parse_args()
        assert args.deterministic is False
        assert args.output_dir == "./experiments/diffusion"

    def test_latent_dim_default_aligned_with_cli(self):
        from eigenforge import train_eigengan

        with mock.patch.object(sys, "argv", ["train_eigengan"]):
            cli_default = train_eigengan.parse_args().z_dim
        assert EigenGenerator().latent_dim == cli_default

    @pytest.mark.parametrize("module_name", ["train_eigengan", "train_diffusion"])
    def test_apply_determinism_sets_full_flag_stack(self, module_name, monkeypatch):
        module = __import__(f"eigenforge.{module_name}", fromlist=["apply_determinism"])
        calls: list[str] = []
        monkeypatch.setattr(
            module.torch, "use_deterministic_algorithms", lambda flag: calls.append("use_det")
        )
        monkeypatch.setattr(module.torch.backends.cudnn, "deterministic", False)
        monkeypatch.setattr(module.torch.backends.cudnn, "benchmark", True)
        module.apply_determinism()
        assert calls == ["use_det"]
        assert module.torch.backends.cudnn.deterministic is True
        assert module.torch.backends.cudnn.benchmark is False

    def test_ddim_rejects_steps_below_two(self):
        scheduler = DiffusionScheduler(num_timesteps=10)

        class _Stub(torch.nn.Module):
            def forward(self, image, t):
                return torch.randn_like(image)

        with pytest.raises(ValueError) as excinfo:
            scheduler.ddim_sample(_Stub(), (1, 3, 8, 8), steps=1)
        assert "at least 2" in str(excinfo.value)
        # steps=2 is the smallest honest grid and must still work.
        out = scheduler.ddim_sample(_Stub(), (1, 3, 8, 8), steps=2)
        assert out.shape == (1, 3, 8, 8)

    def test_cpu_runs_are_bitwise_reproducible_under_same_seed(self):
        def _tiny_run(seed: int) -> list[float]:
            torch.manual_seed(seed)
            generator = EigenGenerator(
                img_size=8, latent_dim=4, n_basis=2, base_channels=8, max_channels=16
            )
            optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
            losses = []
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = generator(torch.randn(2, 4)).square().mean()
                loss = loss + generator.total_orthogonality_loss()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            return losses

        assert _tiny_run(0) == _tiny_run(0)

    def test_lazy_r1_skips_autograd_graph_on_non_regularization_steps(self):
        from eigenforge.models.eigengan import EigenDiscriminator
        from eigenforge.train_eigengan import discriminator_loss_with_lazy_r1

        torch.manual_seed(0)
        disc = EigenDiscriminator(img_size=8, base_channels=8).eval()
        gen = EigenGenerator(
            img_size=8, latent_dim=4, n_basis=2, base_channels=8, max_channels=16
        ).eval()
        calls: list[bool] = []
        orig_forward = disc.forward

        def spy_forward(x):
            calls.append(bool(x.requires_grad))
            return orig_forward(x)

        disc.forward = spy_forward  # type: ignore[method-assign]
        real = torch.randn(2, 3, 8, 8)
        z = torch.randn(2, 4)

        loss_non_reg, _ = discriminator_loss_with_lazy_r1(
            disc, gen, real, z, step=1, d_reg_every=16, r1_gamma=10.0
        )
        assert calls == [False, False]  # real + fake only; no R1 second-order pass
        assert torch.isfinite(loss_non_reg)

        calls.clear()
        loss_reg, _ = discriminator_loss_with_lazy_r1(
            disc, gen, real, z, step=16, d_reg_every=16, r1_gamma=10.0
        )
        assert calls == [False, False, True]  # 3rd pass has real.requires_grad=True
        assert torch.isfinite(loss_reg)

        calls.clear()
        loss_zero_gamma, _ = discriminator_loss_with_lazy_r1(
            disc, gen, real, z, step=16, d_reg_every=16, r1_gamma=0.0
        )
        assert calls == [False, False], "r1_gamma=0 must never build the R1 graph"
        assert loss_zero_gamma.item() == pytest.approx(loss_non_reg.item(), rel=1e-5)

        loss_small_gamma, _ = discriminator_loss_with_lazy_r1(
            disc, gen, real, z, step=16, d_reg_every=16, r1_gamma=1e-6
        )
        assert loss_small_gamma.item() == pytest.approx(loss_non_reg.item(), rel=0.10)
        loss_reg_half_period, _ = discriminator_loss_with_lazy_r1(
            disc, gen, real, z, step=16, d_reg_every=8, r1_gamma=10.0
        )
        # Lazy R1 scales gamma by d_reg_every (16 vs 8 => exact 2:1 R1 increment).
        assert (loss_reg - loss_non_reg).item() == pytest.approx(
            (loss_reg_half_period - loss_non_reg).item() * 2.0, rel=1e-4
        )


# --------------------------- review wave-8 gap coverage ------------------------


class TestCheckpointEdgeBranches:
    """Branches the first pass left untested (pr-test-analyzer C1-C3/C8)."""

    def test_non_dict_payload_is_schema_error(self, tmp_path):
        artifact = tmp_path / "raw.pt"
        durable_replace_bytes(artifact, lambda handle: torch.save(torch.ones(3), handle))
        with pytest.raises(CheckpointSchemaError) as excinfo:
            load_checkpoint(artifact, required_keys=_REQUIRED)
        assert "Tensor" in str(excinfo.value)

    @pytest.mark.parametrize("bad_epoch", ["3", True, -1])
    def test_invalid_epoch_values_are_rejected(self, tmp_path, bad_epoch):
        artifact = tmp_path / "latest.pt"
        payload = _tiny_payload()
        payload["epoch"] = bad_epoch
        durable_replace_bytes(artifact, lambda handle: torch.save(payload, handle))
        with pytest.raises(CheckpointSchemaError) as excinfo:
            load_checkpoint(artifact, required_keys=_REQUIRED)
        assert "epoch" in str(excinfo.value)

    def test_manifest_path_that_is_not_a_file_fails_closed(self, tmp_path):
        _save(tmp_path)
        # An operator accident (directory at the sidecar path) must not
        # silently degrade to an unverified resume.
        (tmp_path / "latest.pt.manifest.json").unlink()
        (tmp_path / "latest.pt.manifest.json").mkdir()
        with pytest.raises(CheckpointIntegrityError) as excinfo:
            load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED)
        assert "not a regular file" in str(excinfo.value)

    def test_require_manifest_refuses_legacy_checkpoint(self, tmp_path):
        artifact = tmp_path / "legacy.pt"
        durable_replace_bytes(artifact, lambda handle: torch.save(_tiny_payload(epoch=1), handle))
        with pytest.raises(CheckpointIntegrityError):
            load_checkpoint(artifact, required_keys=_REQUIRED, require_manifest=True)

    def test_manifest_write_failure_fails_closed_without_fallback(self, tmp_path, monkeypatch):
        """A failed manifest write must fail closed on the NEXT strict resume.

        The checkpoint bytes are new while the manifest still describes the
        previous artifact, so sha verification rejects rather than silently
        resuming from unreferenced weights. (Recovery from the retained previous
        generation is covered by ``TestCrashConsistentCommit``.)
        """
        from eigenforge import artifacts as artifacts_module

        _save(tmp_path)

        def _boom(self, path):
            raise OSError("disk full")

        # mutate the artifact afterwards
        monkeypatch.setattr(artifacts_module.ArtifactManifest, "to_json", _boom)
        with pytest.raises(OSError):
            _save(tmp_path)
        with pytest.raises(CheckpointIntegrityError):
            load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED)

    def test_json_roundtrip_config_values_do_not_phantom_drift(self, tmp_path):
        """tuple/list asymmetry across the JSON manifest must not read as drift."""
        save_checkpoint(
            tmp_path / "tuple.pt",
            _tiny_payload(),
            model_kind="eigengan",
            seed=0,
            step=1,
            training_config={"shape": (64, 64)},
        )
        state = load_checkpoint(
            tmp_path / "tuple.pt",
            required_keys=_REQUIRED,
            expected_config={"shape": (64, 64)},
        )
        assert state.manifest is not None

    def test_legacy_resume_emits_loud_warning(self, tmp_path):
        from eigenforge.common.logger import logger

        artifact = tmp_path / "legacy.pt"
        durable_replace_bytes(artifact, lambda handle: torch.save(_tiny_payload(epoch=2), handle))
        captured: list[str] = []
        handler_id = logger.add(
            lambda message: captured.append(message.record["message"]), level="WARNING"
        )
        try:
            state = load_checkpoint(artifact, required_keys=_REQUIRED)
        finally:
            logger.remove(handler_id)
        assert state.manifest is None
        assert any("no sibling manifest" in message for message in captured)


class TestResumeEmaFallback:
    """d9f7df4's contract: pre-EMA checkpoints fall back with a loud warning."""

    def test_missing_ema_generator_warns_and_seeds_from_raw(self, tmp_path):
        from eigenforge.common.logger import logger
        from eigenforge.train_eigengan import apply_resume_state

        generator = EigenGenerator(img_size=8, latent_dim=4, n_basis=2, base_channels=8, max_channels=16)
        ema_generator = copy.deepcopy(generator)
        # Push EMA away from the raw weights so the fallback is observable.
        with torch.no_grad():
            for parameter in ema_generator.parameters():
                parameter.add_(0.5)
        discriminator = EigenDiscriminator(img_size=8, base_channels=16)
        g_optim = torch.optim.Adam(generator.parameters(), lr=1e-4)
        d_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        artifact = tmp_path / "no_ema.pt"
        payload = _tiny_payload(epoch=9)
        del payload["ema_generator"]
        durable_replace_bytes(artifact, lambda handle: torch.save(payload, handle))
        state = load_checkpoint(artifact, required_keys=_REQUIRED)

        captured: list[str] = []
        handler_id = logger.add(
            lambda message: captured.append(message.record["message"]), level="WARNING"
        )
        try:
            start_epoch = apply_resume_state(
                generator=generator,
                ema_generator=ema_generator,
                discriminator=discriminator,
                g_optim=g_optim,
                d_optim=d_optim,
                resume_state=state,
                path=str(artifact),
            )
        finally:
            logger.remove(handler_id)

        assert start_epoch == 9
        assert any("lacks ema_generator" in message for message in captured)
        for ema_parameter, raw_parameter in zip(ema_generator.parameters(), generator.parameters()):
            assert torch.equal(ema_parameter, raw_parameter)


# ------------------- crash-consistent commit / single-read verification ------------------


def _capture_warnings():
    from eigenforge.common.logger import logger

    captured: list[str] = []
    handler_id = logger.add(lambda message: captured.append(message.record["message"]), level="WARNING")
    return captured, lambda: logger.remove(handler_id)


class TestSingleReadVerification:
    def test_file_swapped_after_hashing_cannot_ride_the_verified_hash(self, tmp_path, monkeypatch):
        """The bytes that were hashed are the bytes that get deserialized.

        The old flow hashed the file, then ``torch.load(path)`` opened it again; a
        swap in between returned the attacker's epoch while reporting "verified".
        """
        from eigenforge import checkpoints as checkpoints_module

        _save(tmp_path)  # epoch 3, manifest describes these bytes
        artifact = tmp_path / "latest.pt"
        evil = tmp_path / "evil.pt"
        durable_replace_bytes(evil, lambda handle: torch.save(_tiny_payload(epoch=999), handle))
        real_load = torch.load

        def swapping_load(source, *args, **kwargs):
            artifact.write_bytes(evil.read_bytes())  # swap lands after the hash was taken
            return real_load(source, *args, **kwargs)

        monkeypatch.setattr(checkpoints_module.torch, "load", swapping_load)
        state = load_checkpoint(artifact, required_keys=_REQUIRED)
        assert state.epoch == 3, "deserialized bytes were not the verified bytes"
        assert state.manifest is not None


class TestCrashConsistentCommit:
    def _bytes(self, tmp_path):
        return (tmp_path / "latest.pt").read_bytes(), (tmp_path / "latest.pt.manifest.json").read_bytes()

    @pytest.mark.parametrize("failure", ["unpicklable_payload", "non_json_config"])
    def test_failures_before_the_disk_phase_leave_the_previous_pair_untouched(self, tmp_path, failure):
        _save(tmp_path)
        before = self._bytes(tmp_path)
        payload = _tiny_payload(epoch=4)
        config = dict(_CONFIG)
        if failure == "unpicklable_payload":
            payload["oops"] = lambda: None
        else:
            config["bad"] = {1, 2, 3}  # a set is not JSON-serializable
        with pytest.raises(Exception):
            save_checkpoint(
                tmp_path / "latest.pt", payload, model_kind="eigengan", seed=1, step=2,
                training_config=config,
            )
        assert self._bytes(tmp_path) == before, "disk changed although the save failed in memory"
        assert not (tmp_path / "latest.pt.prev").exists(), "a failed save must not rotate generations"
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []

    def test_previous_generation_is_retained_and_independently_verifiable(self, tmp_path):
        _save(tmp_path, payload=_tiny_payload(epoch=3))
        _save(tmp_path, payload=_tiny_payload(epoch=4))
        prev = tmp_path / "latest.pt.prev"
        assert prev.is_file() and (tmp_path / "latest.pt.prev.manifest.json").is_file()
        assert load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED, require_manifest=True).epoch == 4
        old = load_checkpoint(prev, required_keys=_REQUIRED, require_manifest=True)
        assert old.epoch == 3 and old.manifest is not None

    def test_interrupted_manifest_write_recovers_from_previous_generation(self, tmp_path, monkeypatch):
        from eigenforge import artifacts as artifacts_module

        _save(tmp_path, payload=_tiny_payload(epoch=3))

        def _boom(self, path):
            raise OSError("disk full")

        with monkeypatch.context() as patch:  # undo only this patch, not the autouse env fixture
            patch.setattr(artifacts_module.ArtifactManifest, "to_json", _boom)
            with pytest.raises(OSError):
                _save(tmp_path, payload=_tiny_payload(epoch=4))
        # New checkpoint + old manifest: strict loading still fails closed ...
        with pytest.raises(CheckpointIntegrityError):
            load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED)
        # ... but resuming with the fallback lands on the intact previous generation.
        captured, stop = _capture_warnings()
        try:
            state = load_checkpoint(
                tmp_path / "latest.pt", required_keys=_REQUIRED, require_manifest=True,
                fallback_to_previous=True,
            )
        finally:
            stop()
        assert state.epoch == 3
        assert any("previous generation" in message for message in captured)

    def test_fallback_covers_a_missing_primary(self, tmp_path):
        _save(tmp_path, payload=_tiny_payload(epoch=3))
        _save(tmp_path, payload=_tiny_payload(epoch=4))
        (tmp_path / "latest.pt").unlink()  # crash between rotation and rename
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED)
        state = load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED, fallback_to_previous=True)
        assert state.epoch == 3

    def test_fallback_does_not_mask_a_config_drift_error(self, tmp_path):
        # Generation 1 matches expected_config (z_dim=999), generation 2 drifts
        # (z_dim=16). If load_checkpoint wrongly caught CheckpointSchemaError and
        # fell back to .prev, this call would silently succeed with epoch=3.
        save_checkpoint(
            tmp_path / "latest.pt",
            _tiny_payload(epoch=3),
            model_kind="eigengan",
            step=30,
            seed=7,
            training_config={**_CONFIG, "z_dim": 999},
            dataset={"name": "celeba", "num_images": 1},
        )
        _save(tmp_path, payload=_tiny_payload(epoch=4))
        with pytest.raises(CheckpointSchemaError):
            load_checkpoint(
                tmp_path / "latest.pt", required_keys=_REQUIRED,
                expected_config={**_CONFIG, "z_dim": 999}, fallback_to_previous=True,
            )

    def test_both_generations_corrupt_reports_the_primary_failure(self, tmp_path):
        _save(tmp_path, payload=_tiny_payload(epoch=3))
        _save(tmp_path, payload=_tiny_payload(epoch=4))
        for name in ("latest.pt", "latest.pt.prev"):
            path = tmp_path / name
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 0xFF
            path.write_bytes(bytes(data))
        with pytest.raises(CheckpointIntegrityError) as excinfo:
            load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED, fallback_to_previous=True)
        assert "latest.pt " in str(excinfo.value) or "latest.pt" in str(excinfo.value)

    def test_filesystem_without_hard_links_degrades_loudly_not_fatally(self, tmp_path, monkeypatch):
        from eigenforge import checkpoints as checkpoints_module

        _save(tmp_path, payload=_tiny_payload(epoch=3))

        def _no_links(src, dst):
            raise OSError("hard links unsupported")

        monkeypatch.setattr(checkpoints_module.os, "link", _no_links)
        captured, stop = _capture_warnings()
        try:
            _save(tmp_path, payload=_tiny_payload(epoch=4))
        finally:
            stop()
        assert any("previous checkpoint generation" in message for message in captured)
        assert not (tmp_path / "latest.pt.prev").exists()
        assert load_checkpoint(tmp_path / "latest.pt", required_keys=_REQUIRED).epoch == 4

    def test_bare_state_dict_loads_for_read_only_consumers(self, tmp_path):
        artifact = tmp_path / "bare.pt"
        generator = EigenGenerator(img_size=8, latent_dim=4, n_basis=2, base_channels=8, max_channels=16)
        durable_replace_bytes(artifact, lambda handle: torch.save(generator.state_dict(), handle))
        with pytest.raises(CheckpointSchemaError):
            load_checkpoint(artifact)  # resume paths keep demanding an epoch
        state = load_checkpoint(artifact, require_epoch=False)
        assert state.epoch == 0 and "initial_const" in state.payload


class TestSelectGeneratorState:
    def test_prefers_ema_without_noise(self):
        captured, stop = _capture_warnings()
        try:
            chosen = select_generator_state({"ema_generator": {"a": 1}, "generator": {"a": 2}}, source="x.pt")
        finally:
            stop()
        assert chosen == {"a": 1} and captured == []

    def test_raw_generator_fallback_is_loud(self):
        captured, stop = _capture_warnings()
        try:
            chosen = select_generator_state({"generator": {"a": 2}}, source="x.pt")
        finally:
            stop()
        assert chosen == {"a": 2}
        assert any("lacks ema_generator" in message for message in captured)

    def test_bare_state_dict_fallback_is_loud(self):
        captured, stop = _capture_warnings()
        try:
            payload = {"weight": 1}
            chosen = select_generator_state(payload, source="x.pt")
        finally:
            stop()
        assert chosen is payload
        assert any("bare generator state_dict" in message for message in captured)


class TestFiniteParameters:
    def _tiny_generator(self):
        return EigenGenerator(img_size=8, latent_dim=4, n_basis=2, base_channels=8, max_channels=16)

    def test_finite_modules_pass_and_integer_buffers_are_ignored(self):
        check_finite_parameters(
            {"generator": self._tiny_generator(), "bn": torch.nn.BatchNorm1d(2)}, epoch=0
        )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_parameter_names_the_tensor_and_epoch(self, bad):
        generator = self._tiny_generator()
        with torch.no_grad():
            generator.initial_const.fill_(bad)
        with pytest.raises(NonFiniteLossError) as excinfo:
            check_finite_parameters({"generator": generator}, epoch=2)
        message = str(excinfo.value)
        assert "generator.initial_const" in message and "epoch=2" in message


class TestEmaWarmupAndResumePolicy:
    def test_ema_decay_warmup_formula(self):
        from eigenforge.train_diffusion import ema_decay_at

        assert ema_decay_at(0, 0.9999) == pytest.approx(0.1)  # (1+0)/(10+0)
        assert ema_decay_at(90, 0.9999) == pytest.approx(0.91)  # (1+90)/(10+90)
        series = [ema_decay_at(n, 0.9999) for n in (0, 10, 100, 1_000, 10_000, 100_000)]
        assert series == sorted(series), "warm-up must be non-decreasing"
        assert ema_decay_at(10_000_000, 0.9999) == 0.9999, "capped at the configured decay"

    def test_warmup_keeps_the_random_init_out_of_the_average(self):
        from eigenforge.train_diffusion import ema_decay_at

        # Constant 0.9999 leaves 0.9999**n of the init after n steps; with the
        # warm-up the init's residual weight after 1000 steps is orders smaller.
        constant_residual = 0.9999 ** 1000
        warm_residual = 1.0
        for step in range(1000):
            warm_residual *= ema_decay_at(step, 0.9999)
        assert constant_residual > 0.9
        assert warm_residual < 1e-2

    @pytest.mark.parametrize(
        "module_name",
        ["train_eigengan", "train_diffusion", "scripts.traverse_eigen_dims"],
    )
    def test_unverified_resume_is_opt_in(self, module_name):
        module = __import__(f"eigenforge.{module_name}", fromlist=["parse_args"])
        with mock.patch.object(sys, "argv", [module_name]):
            assert module.parse_args().allow_unverified_resume is False
        with mock.patch.object(sys, "argv", [module_name, "--allow-unverified-resume"]):
            assert module.parse_args().allow_unverified_resume is True


class TestGitMetadata:
    def test_resolves_the_repo_root_independent_of_cwd(self, tmp_path, monkeypatch):
        import shutil

        from eigenforge.artifacts import _REPO_ROOT, git_metadata

        if shutil.which("git") is None or not (_REPO_ROOT / ".git").exists():
            pytest.skip("git checkout is not available")
        monkeypatch.chdir(tmp_path)  # not a git repository
        meta = git_metadata()
        assert meta is not None and len(meta["commit"]) == 40, (
            "git metadata must come from the repo that contains the code, not the CWD"
        )

    def test_load_checkpoint_enforces_torch_guard_before_reading(self, tmp_path, monkeypatch):
        from eigenforge.common.torch_guard import UnsafeTorchError

        monkeypatch.delenv("EIGENFORGE_ALLOW_UNSAFE_TORCH", raising=False)
        monkeypatch.setattr(torch, "__version__", "2.5.1+cu121")
        with pytest.raises(UnsafeTorchError):
            load_checkpoint(tmp_path / "does_not_even_exist.pt")

