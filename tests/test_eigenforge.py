import io
import logging
import math
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from eigenforge.evaluation.evaluator import GenAIEvaluator
from eigenforge.models.diffusion_scheduler import DiffusionScheduler
from eigenforge.models.eigengan import EigenDiscriminator, EigenGenerator, SubspaceLayer


def test_subspace_layer_orthogonality():
    layer = SubspaceLayer(dim=32, n_basis=4)
    loss = layer.orthogonality_loss()
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_ortho_loss_penalizes_norm_drift():
    layer = SubspaceLayer(dim=16, n_basis=4, orthonormal_init=False)
    layer.basis.data *= 3.0
    assert layer.orthogonality_loss().item() > 1.0


def test_eigengan_forward_and_control():
    model = EigenGenerator(z_dim=32, n_basis=4)
    z = torch.randn(2, 32)
    # 标准前向
    img = model(z)
    assert img.shape == (2, 3, 64, 64)

    # 潜空间语义操纵
    perturbed = model.sample_with_eigen_control(z, layer_idx=0, basis_idx=0, alpha=2.0)
    assert perturbed.shape == (2, 3, 64, 64)

    # 正交性正则项
    ortho_loss = model.total_orthogonality_loss()
    assert ortho_loss.item() >= 0.0


def test_diffusion_scheduler():
    scheduler = DiffusionScheduler(timesteps=100)
    x_start = torch.randn(2, 3, 64, 64)
    t = torch.tensor([10, 50])
    noise = torch.randn_like(x_start)

    x_noisy = scheduler.q_sample(x_start, t, noise)
    assert x_noisy.shape == x_start.shape

    # 步进降噪
    pred_noise = torch.randn_like(x_start)
    prev_sample = scheduler.p_sample(pred_noise, x_noisy, t=50)
    assert prev_sample.shape == x_start.shape


def test_ddim_time_grid_descending():
    scheduler = DiffusionScheduler(timesteps=100)
    grid = scheduler._build_time_grid(steps=10)
    assert grid[0].item() == 99
    assert grid[-2].item() == 0
    assert grid[-1].item() == -1
    assert torch.all(grid[:-1] > grid[1:])


def test_ddim_sampling_uses_descending_grid():
    scheduler = DiffusionScheduler(timesteps=100)

    class RecordingZeroUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x, t):
            self.seen.append(t.detach().cpu())
            return torch.zeros_like(x)

    model = RecordingZeroUNet()
    out = scheduler.ddim_sample(model, (1, 3, 16, 16), steps=10)
    seen = [int(step.item()) for step in model.seen]
    assert seen[0] == 99
    assert all(a > b for a, b in zip(seen, seen[1:]))
    assert out.shape == (1, 3, 16, 16)
    assert torch.isfinite(out).all()


def test_genai_evaluator():
    x = torch.randn(4, 3, 64, 64)
    y = torch.randn(4, 3, 64, 64)
    psnr = GenAIEvaluator.calculate_psnr(x, y)
    assert psnr > 0.0

    lpips_proxy = GenAIEvaluator.calculate_lpips_proxy(x, y)
    assert lpips_proxy >= 0.0


# ------------------------------------------------ schedule scaling and x0 clipping


def test_linear_schedule_at_t1000_is_the_unscaled_ho_et_al_schedule():
    scheduler = DiffusionScheduler(num_timesteps=1000, schedule_type="linear")
    expected = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float32)
    assert torch.equal(scheduler.betas, expected)


def test_linear_schedule_rescales_with_t_so_the_terminal_state_is_noise():
    # Reference codebases (improved-diffusion, guided-diffusion, DiT) scale both
    # endpoints by 1000/T. Unscaled, T=100 ends at alpha_bar_T ~ 0.36 and sampling
    # from N(0, I) would start off-distribution.
    terminal = {
        t: DiffusionScheduler(num_timesteps=t, schedule_type="linear").alphas_cumprod[-1].item()
        for t in (100, 250, 500, 1000)
    }
    assert all(value < 1e-3 for value in terminal.values()), terminal
    ratio = max(terminal.values()) / min(terminal.values())
    assert ratio < 10, f"terminal SNR should be similar across T, got {terminal}"


def test_tiny_timestep_schedules_stay_finite_and_sampleable():
    class Rand(torch.nn.Module):
        def forward(self, x, t):
            return torch.randn_like(x)

    for t in (2, 4, 10, 50):
        scheduler = DiffusionScheduler(num_timesteps=t, schedule_type="linear")
        assert torch.all(scheduler.betas <= 0.9999) and torch.isfinite(scheduler.alphas_cumprod).all()
        out = scheduler.ddim_sample(Rand(), (1, 3, 8, 8), steps=min(t, 5), eta=1.0)
        assert torch.isfinite(out).all()


def test_ddim_x0_clipping_bounds_the_first_step_error_amplification():
    scheduler = DiffusionScheduler(num_timesteps=1000, schedule_type="cosine")

    class SmallBias(torch.nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, 1e-3)  # a 1e-3 epsilon error

    torch.manual_seed(1)
    unclipped = scheduler.ddim_sample(SmallBias(), (1, 3, 8, 8), steps=10, clip_denoised=False)
    torch.manual_seed(1)
    clipped = scheduler.ddim_sample(SmallBias(), (1, 3, 8, 8), steps=10)  # default: clip
    assert clipped.abs().max().item() <= 1.0 + 1e-6
    assert unclipped.abs().max().item() > 100 * clipped.abs().max().item()


def test_ddim_clipping_is_inert_when_x0_prediction_is_already_in_range():
    scheduler = DiffusionScheduler(num_timesteps=100, schedule_type="cosine")
    target = torch.full((1, 3, 8, 8), 0.25)

    class Oracle(torch.nn.Module):
        """epsilon that makes x0 == target at every step (no clipping needed)."""

        def __init__(self, sched):
            super().__init__()
            self.sched = sched

        def forward(self, x, t):
            ab = self.sched.alphas_cumprod[t.long()].view(-1, 1, 1, 1)
            return (x - ab.sqrt() * target) / (1 - ab).sqrt()

    torch.manual_seed(3)
    a = scheduler.ddim_sample(Oracle(scheduler), (1, 3, 8, 8), steps=10, clip_denoised=False)
    torch.manual_seed(3)
    b = scheduler.ddim_sample(Oracle(scheduler), (1, 3, 8, 8), steps=10, clip_denoised=True)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-3)


# ----------------------------- extended metrics (wave 2) ---------------------


def test_ssim_is_one_for_identical_images():
    x = torch.rand(2, 3, 64, 64)
    assert GenAIEvaluator.calculate_ssim(x, x) > 0.99


def test_ssim_lower_for_different_images():
    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64)
    assert GenAIEvaluator.calculate_ssim(x, y) < 0.9


def test_kid_identical_distribution_is_not_judged_worse_than_a_different_one():
    torch.manual_seed(0)
    real = torch.randn(512, 64)
    different = torch.randn(512, 64) * 2.0 + 1.0
    identical, _ = GenAIEvaluator.calculate_kid(
        real, real, subsets=20, max_subset_size=128, estimator="biased"
    )
    shifted, _ = GenAIEvaluator.calculate_kid(
        real, different, subsets=20, max_subset_size=128, estimator="biased"
    )
    assert identical <= shifted


def test_kid_unbiased_matches_reference_and_clips_negative_mean():
    torch.manual_seed(0)
    feats = torch.randn(256, 64)
    raw, _ = GenAIEvaluator.calculate_kid(
        feats, feats, subsets=20, max_subset_size=128, clip_negative=False
    )
    assert raw < 0.0 and raw > -1.0
    clipped, _ = GenAIEvaluator.calculate_kid(feats, feats, subsets=20, max_subset_size=128)
    assert clipped == 0.0


def test_kid_matches_reference_closed_form():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mean, _ = GenAIEvaluator.calculate_kid(x, x, subsets=1, clip_negative=False)
    assert abs(mean - (-2.375)) < 1e-6

    y = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    biased, _ = GenAIEvaluator.calculate_kid(x, y, subsets=1, estimator="biased")
    assert abs(biased - 0.5625) < 1e-6


def test_kid_is_positive_for_distinct_distributions():
    torch.manual_seed(0)
    real = torch.randn(512, 64)
    fake = torch.randn(512, 64) * 1.5 + 0.5
    for estimator in ("unbiased", "biased"):
        mean, _ = GenAIEvaluator.calculate_kid(
            real, fake, subsets=20, max_subset_size=256, estimator=estimator
        )
        assert mean > 0.0, f"{estimator} KID must be positive for distinct distributions"


def test_kid_rejects_unknown_estimator():
    feats = torch.randn(8, 4)
    with pytest.raises(ValueError):
        GenAIEvaluator.calculate_kid(feats, feats, subsets=1, estimator="bogus")


@pytest.mark.slow
def test_inception_score_positive(monkeypatch):
    pytest.importorskip("torchvision")
    monkeypatch.setenv("EIGENFORGE_ALLOW_UNSAFE_TORCH", "1")
    images = torch.rand(16, 3, 128, 128)
    mean, std = GenAIEvaluator.calculate_inception_score(images, splits=2)
    assert mean >= 1.0 and std >= 0.0


def test_ppl_per_style_layer_keys():
    gen = EigenGenerator(img_size=32, latent_dim=64, n_basis=4)
    evaluator = GenAIEvaluator(device="cpu")
    per_layer = evaluator.compute_ppl(gen, num_samples=4, per_style_layer=True)
    assert set(per_layer.keys()) == set(range(gen.n_stages))


# ---------------------- EigenGAN & scheduler guards (wave 3/4/6) -------------


def test_orthogonality_loss_pinned_fixed_matrix():
    layer = SubspaceLayer(dim=4, n_basis=2, orthonormal_init=False)
    layer.basis.data = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    loss_normalized = layer.orthogonality_loss(scale="normalized").item()
    loss_sum_half = layer.orthogonality_loss(scale="sum_half").item()
    assert loss_normalized == pytest.approx(2.25)
    assert loss_sum_half == pytest.approx(4.5)


def test_eigengan_supports_non_default_base_channels():
    model = EigenGenerator(img_size=64, latent_dim=32, n_basis=4, base_channels=16)
    out = model(torch.randn(2, 32))
    assert out.shape == (2, 3, 64, 64)
    assert model.initial_channels == 256


def test_subspace_layer_validates_dim():
    with pytest.raises(ValueError, match="cannot exceed out_dim"):
        SubspaceLayer(dim=4, n_basis=8)


def test_ppl_per_style_layer_isolates_target_layer():
    gen = EigenGenerator(img_size=32, latent_dim=64, n_basis=4)
    evaluator = GenAIEvaluator(device="cpu")
    per_layer = evaluator.compute_ppl(gen, num_samples=4, per_style_layer=True)
    assert set(per_layer.keys()) == set(range(gen.n_stages))
    assert all(math.isfinite(v) and v >= 0.0 for v in per_layer.values())


def test_scheduler_rejects_out_of_range_timesteps():
    sched = DiffusionScheduler(timesteps=100)
    x = torch.randn(1, 3, 16, 16)
    with pytest.raises(ValueError, match="out of range"):
        sched.q_sample(x, torch.tensor([100]))
    with pytest.raises(ValueError, match="out of range"):
        sched.q_sample(x, torch.tensor([-1]))
    with pytest.raises(ValueError, match="scalar or shape"):
        sched.q_sample(torch.randn(2, 3, 16, 16), torch.tensor([10]))


def test_ddim_rejects_bad_args():
    sched = DiffusionScheduler(timesteps=100)
    model = torch.nn.Identity()
    with pytest.raises(ValueError, match="eta"):
        sched.ddim_sample(model, (1, 3, 8, 8), steps=10, eta=-0.5)
    with pytest.raises(ValueError, match="shape"):
        sched.ddim_sample(model, (1, 3), steps=10)


def test_logger_file_sink_is_configurable(monkeypatch, tmp_path):
    from eigenforge.common import logger as logger_module
    from eigenforge.common.config import AppConfig

    monkeypatch.setenv("LOG_FILE", "")
    disabled = AppConfig()
    assert disabled.log_file == ""

    monkeypatch.setenv("LOG_FILE", str(tmp_path / "custom.log"))
    relocated = AppConfig()
    assert relocated.log_file == str(tmp_path / "custom.log")

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler_id = logger_module.logger.add(handler, level="INFO")
    try:
        logger_module.logger.info("sink-config smoke")
    finally:
        logger_module.logger.remove(handler_id)
    assert "sink-config smoke" in stream.getvalue()


def test_eigengan_generator_backward_through_resblocks():
    torch.manual_seed(0)
    generator = EigenGenerator(img_size=32, latent_dim=32, n_basis=4, base_channels=16, max_channels=64)
    out = generator(generator.sample_latents(2))
    out.mean().backward()
    grads = [p.grad for p in generator.parameters() if p.grad is not None]
    assert grads, "no gradients reached any parameter"
    assert all(torch.isfinite(g).all() for g in grads)


def test_ddim_eta_changes_sampling():
    class _ZeroModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    scheduler = DiffusionScheduler(num_timesteps=50, schedule_type="linear")
    torch.manual_seed(123)
    deterministic = scheduler.ddim_sample(_ZeroModel(), (1, 3, 8, 8), steps=10, eta=0.0)
    torch.manual_seed(123)
    stochastic = scheduler.ddim_sample(_ZeroModel(), (1, 3, 8, 8), steps=10, eta=1.0)
    assert (deterministic - stochastic).abs().max().item() > 1e-4


def test_q_sample_accepts_scalar_timestep_tensor():
    scheduler = DiffusionScheduler(num_timesteps=100, schedule_type="linear")
    x = torch.randn(2, 3, 8, 8)
    out = scheduler.q_sample(x, torch.tensor(5))
    assert out.shape == x.shape
    out_batch = scheduler.q_sample(x, torch.tensor([5, 9]))
    assert out_batch.shape == x.shape


def test_p_sample_accepts_scalar_timestep_int():
    scheduler = DiffusionScheduler(num_timesteps=100, schedule_type="linear")
    x = torch.randn(2, 3, 8, 8)
    out = scheduler.p_sample(torch.zeros_like(x), x, 7)
    assert out.shape == x.shape


def test_lazy_r1_only_builds_graph_on_scheduled_steps(monkeypatch):
    from eigenforge import train_eigengan as train_module

    torch.manual_seed(0)
    generator = EigenGenerator(img_size=32, latent_dim=32, n_basis=4, base_channels=16, max_channels=64)
    discriminator = EigenDiscriminator(img_size=32, base_channels=16)
    real = torch.randn(2, 3, 32, 32)

    calls = {"r1": 0}
    real_r1 = train_module.r1_penalty

    def counting_r1(real_logits, real_images):
        calls["r1"] += 1
        return real_r1(real_logits, real_images)

    monkeypatch.setattr(train_module, "r1_penalty", counting_r1)

    for step in range(4):
        z = torch.randn(2, 32)
        loss, _ = train_module.discriminator_loss_with_lazy_r1(
            discriminator, generator, real, z, r1_gamma=10.0, d_reg_every=16, step=step
        )
        loss.backward()

    assert calls["r1"] == 1

    calls["r1"] = 0
    loss, _ = train_module.discriminator_loss_with_lazy_r1(
        discriminator, generator, real, torch.randn(2, 32), r1_gamma=0.0, d_reg_every=16, step=0
    )
    loss.backward()
    assert calls["r1"] == 0

    calls["r1"] = 0
    for step in range(3):
        loss, _ = train_module.discriminator_loss_with_lazy_r1(
            discriminator, generator, real, torch.randn(2, 32), r1_gamma=10.0, d_reg_every=1, step=step
        )
        loss.backward()
    assert calls["r1"] == 3


# --------------------- reference config mirror guard -------------------------


def test_eigenforge_mirror_stays_non_authoritative_and_matches_cli_defaults():
    from eigenforge import train_eigengan

    pkg_root = Path(__file__).resolve().parents[1]
    mirror_path = pkg_root / "configs" / "eigenforge.yaml"
    text = mirror_path.read_text(encoding="utf-8")
    assert "不被代码加载" in text

    excluded = {"tests", "__pycache__", "outputs", "logs", ".zcode", ".pytest_cache", ".ruff_cache"}
    sources = [
        p
        for p in pkg_root.rglob("*.py")
        if not (excluded & set(p.relative_to(pkg_root).parts)) and p.name != "conftest.py"
    ]
    offenders = [
        p.relative_to(pkg_root).as_posix()
        for p in sources
        if re.search(r"configs/|eigenforge\.yaml|yaml\.safe_load", p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"configs/eigenforge.yaml must not be loaded by runtime code: {offenders}"

    def _scalar(key: str) -> str:
        match = re.search(rf"^\s*{re.escape(key)}\s*:\s*([^#\n]+)", text, re.MULTILINE)
        assert match, f"{key} missing from mirror"
        return match.group(1).strip()

    with patch.object(sys, "argv", ["train_eigengan.py"]):
        defaults = vars(train_eigengan.parse_args())

    assert float(_scalar("ortho_weight")) == defaults["ortho_weight"]
    assert float(_scalar("r1_gamma")) == defaults["r1_gamma"]
    assert int(_scalar("d_reg_every")) == defaults["d_reg_every"]
    assert int(_scalar("n_basis")) == defaults["n_basis"]
    assert int(_scalar("z_dim")) == defaults["z_dim"]
    assert int(_scalar("image_size")) == defaults["image_size"]
    assert int(_scalar("batch_size")) == defaults["batch_size"]
    assert float(_scalar("lr")) == defaults["lr"]
    assert float(_scalar("ema_decay")) == defaults["ema_decay"]
    assert int(_scalar("seed")) == defaults["seed"]


# ---------------- paper metrics (Table 1, Appendix A, Tables 2-3) ------------


def test_entropy_coefficient_closed_form_bounds():
    from eigenforge.evaluation.entropy_coefficient import binary_entropy, entropy_coefficient

    # Constant p=0.5 across all bins -> zero mutual information -> U(Y|Z)=0
    constant = torch.full((10, 2), 0.5)
    assert torch.allclose(entropy_coefficient(constant), torch.zeros(2, dtype=torch.float64), atol=1e-9)

    # Deterministic attribute per bin (half 0, half 1) -> H(Y|Z)=0 -> U(Y|Z)=1
    deterministic = torch.tensor([[0.0], [1.0]] * 5, dtype=torch.float32)
    assert torch.allclose(entropy_coefficient(deterministic), torch.ones(1, dtype=torch.float64), atol=1e-9)
    assert binary_entropy(torch.tensor([0.0, 1.0])).abs().max().item() == 0.0


def test_linear_case_mle_recovers_pca_subspace():
    from eigenforge.evaluation.linear_case_mle import linear_case_mle

    torch.manual_seed(0)
    samples = torch.randn(256, 6) * torch.tensor([5.0, 3.0, 0.2, 0.2, 0.2, 0.2])
    fit = linear_case_mle(samples, subspace_dim=2)
    assert fit.basis.shape == (6, 2)
    assert torch.allclose(fit.basis.t() @ fit.basis, torch.eye(2), atol=1e-5)
    assert fit.sigma_sq.item() >= 0.0
    assert (fit.importance[0] >= fit.importance[1] >= 0.0).item()


def test_toy_basis_similarity_losses_and_cosine():
    from eigenforge.evaluation.linear_case_mle import linear_case_mle
    from eigenforge.evaluation.toy_basis_similarity import (
        LOSSES,
        basis_similarity,
        get_adversarial_losses,
    )

    mle = linear_case_mle(torch.randn(64, 4), subspace_dim=2)
    sim = basis_similarity(mle.basis.t(), mle.basis)
    assert sim == pytest.approx(1.0, abs=1e-5)

    logits = torch.randn(8, 1)
    # The model card documents exactly five losses; pin the count so a sixth
    # loss cannot be added while the documentation still says five (the card
    # itself used to claim six).
    assert len(LOSSES) == 5
    for mode in LOSSES:
        d_fn, g_fn = get_adversarial_losses(mode)
        assert torch.isfinite(d_fn(logits, logits))
        assert torch.isfinite(g_fn(logits))



# ------------------------------------------------ QR basis continuity


def test_orthonormal_basis_is_continuous_across_a_zero_crossing():
    # Unnormalized Householder QR flips a column of Q (jump ~1.2) when one basis
    # entry crosses zero; diag(R) >= 0 keeps the basis, and so the axis, stable.
    torch.manual_seed(0)
    layer = SubspaceLayer(dim=16, n_basis=4)
    base = layer.basis.detach().clone()
    previous, worst = None, 0.0
    for value in torch.linspace(-1e-3, 1e-3, 201):
        candidate = base.clone()
        candidate[0, 0] = value
        layer.basis.data = candidate
        q = layer.get_orthonormal_basis().detach()
        if previous is not None:
            worst = max(worst, (q - previous).abs().max().item())
        previous = q
    assert worst < 1e-2, f"orthonormal basis jumped by {worst:.3f} across a tiny basis change"


def test_orthonormal_basis_is_orthonormal_and_spans_the_same_subspace():
    torch.manual_seed(1)
    layer = SubspaceLayer(dim=16, n_basis=4, orthonormal_init=False)
    q = layer.get_orthonormal_basis().detach()
    torch.testing.assert_close(q @ q.t(), torch.eye(4), atol=1e-5, rtol=0)
    # Same row space as the raw basis: projecting the raw rows onto Q loses nothing.
    raw = layer.basis.detach()
    torch.testing.assert_close(raw @ q.t() @ q, raw, atol=1e-4, rtol=0)


def test_orthogonality_loss_default_scale_is_normalized_gram_mse():
    """The DEFAULT scale is the training baseline, and only a closed form pins it.

    AGENTS.md hard constraint: the normalized Gram-MSE is the deliberate
    training baseline. The only default-path assertion above is `> 1.0`, which
    both scales satisfy (measured: 106.64 normalized vs 853.09 sum_half, a
    factor K^2/2 = 8), so flipping the default would leave the suite green.
    """
    layer = SubspaceLayer(dim=4, n_basis=2, orthonormal_init=False)
    layer.basis.data = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    # gram = diag(1, 4) => normalized = MSE(gram, I) = 9/4 = 2.25; sum_half = 4.5
    assert layer.orthogonality_loss().item() == pytest.approx(2.25)


def test_total_orthogonality_loss_keeps_normalized_scale():
    """The aggregate API the trainer calls keeps the same scale.

    total = mean_l(sum_half_l) * 2 / K^2 (measured to 1e-12). If the aggregate
    silently switched to the sum_half convention the per-layer test above would
    still pass while every training run moved by a factor K^2/2.
    """
    torch.manual_seed(0)
    model = EigenGenerator(img_size=8, z_dim=16, n_basis=4, base_channels=8, max_channels=16)
    for layer in model.subspace_layers:
        layer.basis.data *= 1.37  # drift off the Stiefel manifold
    total = model.total_orthogonality_loss().item()
    sum_half = torch.stack(
        [layer.orthogonality_loss(scale="sum_half") for layer in model.subspace_layers]
    ).mean().item()
    assert total == pytest.approx(2 * sum_half / 4 ** 2)


def test_cli_reference_table4_covers_every_train_diffusion_flag():
    """表 4 claims to be the complete parameter list — hold it to that.

    Before the fix it omitted --device/--seed/--num-workers/--no-amp/
    --deterministic, all of which train_diffusion.py really accepts. The
    section must be sliced by its own heading, or the shared flags (which also
    appear in 表 2 for train_eigengan) would make the test pass vacuously.
    """
    from eigenforge import train_diffusion

    doc = (Path(__file__).resolve().parents[1] / "docs" / "CLI_AND_CONFIGURATION_REFERENCE.md").read_text(
        encoding="utf-8"
    )
    section = doc.split("### 2.3")[1].split("## 3.")[0]
    with patch.object(sys, "argv", ["train_diffusion.py"]):
        dests = vars(train_diffusion.parse_args())
    missing = [
        f"--{d}".replace("_", "-")
        for d in dests
        if f"--{d}".replace("_", "-") not in section
    ]
    assert not missing, f"表 4 missing train_diffusion flags: {missing}"
