"""Tiny-scale smoke regression for the official-layout EigenGAN
(``models/eigengan_faithful.py``).

The module is a line-by-line PyTorch re-expression of the TensorFlow original
and, before this file existed, had never been imported or executed. These tests
pin the operator wiring and numerical health at a tiny scale (8px, one layer,
small ``eps_dim`` — shape wiring is what is under test, not the official
parameter count) so a future edit cannot silently break the forward pass, the
gradient flow, the traversal, or the orthogonality term.

This is deliberately NOT a parity or quality claim: the full-network
layer-by-layer TensorFlow comparison is still open (see the model card), and
nothing here trains the model. No checkpoint is loaded, so the torch
version floor in ``common/torch_guard.py`` is irrelevant to this file.
"""

from __future__ import annotations

import math

import pytest
import torch

from eigenforge.models.eigengan_faithful import (
    LEAKY_SLOPE,
    EigenGANDiscriminator,
    EigenGANGenerator,
    SameStride2Conv2d,
    SameStride2ConvTranspose2d,
    gram_schmidt,
    hinge_v1_d_loss,
    hinge_v1_g_loss,
    official_init_,
    orthogonal_regularizer,
    zero_centered_gradient_penalty,
)

# One subspace layer -> 4 * 2**1 = 8px images; small eps keeps the smoke fast.
Z_DIMS = (6,)
EPS_DIM = 32
IMAGE_SIZE = 8
BATCH = 2


@pytest.fixture(autouse=True)
def _seed_each_test():
    torch.manual_seed(0)


@pytest.fixture()
def generator() -> EigenGANGenerator:
    return EigenGANGenerator(z_dims=Z_DIMS, eps_dim=EPS_DIM)


@pytest.fixture()
def discriminator() -> EigenGANDiscriminator:
    return EigenGANDiscriminator(image_size=IMAGE_SIZE)


# ------------------------------------------------------------- forward wiring


class TestForward:
    def test_output_shape_range_and_finiteness(self, generator):
        zs, eps = generator.sample_latents(BATCH)
        image = generator(zs, eps)
        assert image.shape == (BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.isfinite(image).all()
        # tanh output head: values must live in [-1, 1].
        assert image.abs().max() <= 1.0

    def test_truncated_sampling_stays_within_bounds(self, generator):
        zs, eps = generator.sample_latents(BATCH, truncation=0.5)
        assert zs[0].abs().max() <= 0.5 + 1e-6
        assert eps.abs().max() <= 0.5 + 1e-6

    def test_wrong_latent_shapes_raise(self, generator):
        zs, eps = generator.sample_latents(BATCH)
        with pytest.raises(ValueError):
            generator(zs[:-1], eps)
        with pytest.raises(ValueError):
            generator(zs, eps[:, :1])

    def test_gram_schmidt_forward_mode_is_finite(self):
        gen = EigenGANGenerator(z_dims=Z_DIMS, eps_dim=EPS_DIM, use_gram_schmidt=True)
        zs, eps = gen.sample_latents(BATCH)
        image = gen(zs, eps)
        assert image.shape == (BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.isfinite(image).all()

    def test_discriminator_scores_have_shape_batch_one(self, generator, discriminator):
        zs, eps = generator.sample_latents(BATCH)
        with torch.no_grad():
            logits = discriminator(generator(zs, eps))
        assert logits.shape == (BATCH, 1)
        assert torch.isfinite(logits).all()


class TestStride2Ops:
    """The SAME-aligned stride-2 pair must compose back to the input size."""

    def test_conv_halves_and_transpose_doubles_even_sizes(self):
        conv = SameStride2Conv2d(3, 4)
        deconv = SameStride2ConvTranspose2d(4, 3)
        x = torch.randn(1, 3, 8, 8)
        half = conv(x)
        assert half.shape == (1, 4, 4, 4)
        assert torch.isfinite(half).all()
        doubled = deconv(half)
        assert doubled.shape == (1, 3, 8, 8)
        assert torch.isfinite(doubled).all()

    def test_same_padding_is_not_symmetric_padding(self):
        """SAME puts the extra pixel bottom/right; a symmetric pad would shift borders."""
        torch.manual_seed(0)
        conv = SameStride2Conv2d(1, 1)
        with torch.no_grad():
            conv.conv.weight.fill_(1.0)
            conv.conv.bias.zero_()
        x = torch.zeros(1, 1, 4, 4)
        x[0, 0, 3, 3] = 1.0  # the bottom-right pixel
        out = conv(x)
        # The single hot pixel lands in the bottom-right output cell (symmetric
        # padding would smear it into the top-left neighbourhood instead).
        assert out[0, 0, 1, 1] > 0
        assert out[0, 0, 0, 0] == 0


# ------------------------------------------------------------------- backward


class TestBackward:
    def test_gradients_reach_every_parameter_group(self, generator):
        zs, eps = generator.sample_latents(BATCH)
        image = generator(zs, eps)
        image.sum().backward()
        checked = {
            "U": generator.U[0],
            "L": generator.L[0],
            "mu": generator.mu[0],
            "fc.weight": generator.fc.weight,
            "to_rgb.weight": generator.to_rgb.weight,
            "up[0].conv.weight": generator.up[0].conv.weight,
        }
        for name, parameter in checked.items():
            assert parameter.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"

    def test_discriminator_backward_is_finite(self, generator, discriminator):
        zs, eps = generator.sample_latents(BATCH)
        logits = discriminator(generator(zs, eps))
        hinge_v1_d_loss(logits, logits.detach() + 1.0).backward()
        assert torch.isfinite(discriminator.features[0].weight.grad).all()


# ------------------------------------------------------------------ traversal


class TestTraverse:
    def test_one_frame_per_value_and_latents_untouched(self, generator):
        zs, eps = generator.sample_latents(BATCH)
        before = zs[0].clone()
        values = [-1.0, 0.0, 1.0]
        frames = list(generator.traverse(zs, eps, layer=0, dim=0, values=values))
        assert len(frames) == len(values)
        for frame in frames:
            assert frame.shape == (BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
            assert torch.isfinite(frame).all()
        # traverse clones the latent list; the caller's tensors must not move.
        assert torch.equal(zs[0], before)

    def test_dims_by_importance_is_a_descending_permutation(self, generator):
        order = generator.dims_by_importance(0)
        magnitudes = generator.L[0].detach().abs()[order]
        assert sorted(order.tolist()) == list(range(generator.L[0].numel()))
        assert torch.equal(magnitudes, magnitudes.sort(descending=True).values)


# ------------------------------------------------------------- orthogonality


class TestOrthogonality:
    def test_loss_is_zero_at_orthogonal_init_and_positive_after_perturbation(self, generator):
        loss = generator.orthogonal_loss()
        assert torch.isfinite(loss) and loss >= 0
        assert loss < 1e-4, "orthogonally initialized rows should give ~0"
        with torch.no_grad():
            generator.U[0].add_(0.3)
        perturbed = generator.orthogonal_loss()
        assert torch.isfinite(perturbed)
        assert perturbed > 0.1

    def test_regularizer_matches_the_official_unscaled_form(self):
        basis = torch.randn(3, 4, 2, 2)
        flat = basis.flatten(1)
        expected = 0.5 * (flat @ flat.t() - torch.eye(3)).square().sum()
        assert float(orthogonal_regularizer(basis)) == pytest.approx(float(expected))

    def test_gram_schmidt_rows_are_orthonormal(self):
        vectors = torch.randn(4, 7)
        basis = gram_schmidt(vectors)
        gram = basis @ basis.t()
        assert torch.allclose(gram, torch.eye(4), atol=1e-5)


# ------------------------------------------------------- losses and init form


class TestLossesAndInit:
    def test_hinge_losses_are_non_negative_and_finite(self):
        real = torch.randn(8, 1)
        fake = torch.randn(8, 1)
        d_loss = hinge_v1_d_loss(real, fake)
        g_loss = hinge_v1_g_loss(fake)
        assert torch.isfinite(d_loss) and d_loss >= 0
        assert torch.isfinite(g_loss) and g_loss >= 0

    def test_zero_centered_gradient_penalty_is_finite_and_non_negative(self, discriminator):
        images = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True)
        logits = discriminator(images)
        penalty = zero_centered_gradient_penalty(logits, images)
        assert torch.isfinite(penalty)
        assert penalty >= 0

    def test_official_init_stays_inside_the_truncated_normal_bounds(self):
        from torch import nn

        module = nn.Linear(8, 4)
        official_init_(module)
        gain = math.sqrt(2.0 / (1.0 + LEAKY_SLOPE**2))
        expected_std = gain / math.sqrt(8) / 0.87962566103423978
        assert module.weight.abs().max() <= 2.0 * expected_std + 1e-6
        assert 0 < module.weight.std() <= expected_std + 1e-9
        assert torch.equal(module.bias, torch.zeros_like(module.bias))


# ----------------------------------------------------------- EMA warm-up protocol


class TestEmaWarmupProtocol:
    """The generator-only EMA must not carry the random init into long averages.

    ``train_diffusion`` already ramps its decay with the TensorFlow
    ``ExponentialMovingAverage(num_updates=...)`` formula; the EigenGAN trainer
    now shares it. This pins the two entries to the same schedule instead of
    asserting a specific implementation detail.
    """

    def test_both_trainers_share_the_warmup_formula(self):
        from eigenforge.train_diffusion import ema_decay_at as diffusion
        from eigenforge.train_eigengan import ema_decay_at as eigengan

        for step in (0, 1, 10, 100, 1_000, 100_000):
            assert eigengan(step, 0.999) == diffusion(step, 0.999)
        assert eigengan(0, 0.999) == pytest.approx(0.1)  # (1+0)/(10+0)
        assert eigengan(10_000_000, 0.999) == 0.999, "capped at the configured decay"
        series = [eigengan(n, 0.999) for n in (0, 10, 100, 1_000)]
        assert series == sorted(series), "warm-up must be non-decreasing"

    def test_warmup_expels_the_random_init_from_the_average(self):
        from eigenforge.train_eigengan import ema_decay_at

        constant_residual = 0.999**1000
        warm_residual = 1.0
        for step in range(1000):
            warm_residual *= ema_decay_at(step, 0.999)
        assert warm_residual < constant_residual / 1000
