"""EigenGAN generator/discriminator laid out like the official TensorFlow code.

PyTorch re-expression of ``module.py``, ``tfprob/gan/loss.py`` (``hinge_v1``) and
``tfprob/gan/gradient_penalty.py`` (``0-gp`` on real samples) from
https://github.com/LynnHo/EigenGAN-Tensorflow at commit
fc3e1a23b84661512ab9a753329fdadb0c96b536 (MIT License, Copyright (c) 2021 Zhenliang He).
Method: He, Kan, Shan, "EigenGAN: Layer-Wise Eigen-Learning for GANs", ICCV 2021
(arXiv:2104.12476).

This module is deliberately separate from ``models/eigengan.py``. That file is the
simplified variant (shared z, channel-only basis) that existing tests, the evaluator
and the traversal script depend on; its ``total_orthogonality_loss`` keeps the
normalized Gram-MSE scale. Nothing here changes it, and the orthogonality term below is
the official un-normalized ``0.5 * sum((UᵀU - I)²)`` summed over layers.

Known deviations from the TensorFlow code (none of them verified numerically; this
module has never been run):

* Stride-2 convolutions reproduce TensorFlow ``SAME`` exactly: the extra pixel of a
  ``SAME`` window goes bottom/right (``pad_total = 1`` for even sizes), and the transposed
  form is the transpose of that same window (full transpose, then drop the last row and
  column). Symmetric ``padding=1`` is *not* equivalent — it shifts the borders by one
  pixel. Tensors are NCHW and transposed weights keep PyTorch's ``[in, out, kh, kw]``
  layout, so weights are still not interchangeable with the TF checkpoints.
* Tensors are NCHW. The flatten order into ``fc`` layers therefore differs from NHWC,
  which only permutes the fully connected units.
* ``weights_regularizer=l2(1.0)`` is omitted: the official ``weight_decay`` default is 0.
"""
import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

ARCHITECTURE_TAG = "eigengan_official_layout"
LEAKY_SLOPE = 0.2
# TF v1.15 variance_scaling("truncated_normal") branch (init_ops.py:522-526), verbatim:
#   # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)  # :523
#   stddev = math.sqrt(scale) / .87962566103423978                               # :524
_TRUNC_STD = 0.87962566103423978


class SameStride2Conv2d(nn.Module):
    """Official 3x3/stride-2 conv under TF ``SAME`` (``tflib.convolution``, padding='SAME').

    ``SAME`` on an even input with k=3, s=2 pads one pixel at the bottom/right only, so
    the symmetric ``padding=1`` equivalent is off by one pixel. ``nn.Conv2d`` has no
    asymmetric padding, hence the explicit ``F.pad`` with ``padding=0``. Source:
    ``tensorflow/core/framework/kernel_shape_util.cc`` — ``padding_before =
    padding_needed / 2`` with the comment "For odd values of total padding, add more
    padding at the 'right' side of the given dimension".
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class SameStride2ConvTranspose2d(nn.Module):
    """Official 3x3/stride-2 transposed conv under TF ``SAME`` (``tflib.dconv2d``).

    ``conv2d_transpose`` is the transpose of the ``SAME`` forward conv — TF's own
    docstring (nn_ops.py:2149-2154): "The transpose of ``conv2d``. This operation is
    sometimes called "deconvolution" after [Deconvolutional Networks], but is
    really the transpose (gradient) of ``conv2d`` rather than an actual
    deconvolution." — i.e. the full transpose of length ``2n + 1`` truncated to
    ``2n``, which is exactly PyTorch's ``ConvTranspose2d(padding=0)`` followed by
    dropping the last row and column.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[..., :-1, :-1]


def gram_schmidt(vectors: torch.Tensor) -> torch.Tensor:
    """Official ``tl.gram_schmidt``: orthonormalize the rows of ``[q, D]`` in order.

    Each row loses its component along the already-built basis, then is normalized.
    Differentiable, so gradients reach the raw ``U`` through the orthonormalization.
    """
    basis = [vectors[0] / vectors[0].norm()]
    for vector in vectors[1:]:
        stacked = torch.stack(basis)
        residual = vector - (stacked @ vector) @ stacked
        basis.append(residual / residual.norm())
    return torch.stack(basis)


def channels(size: int, dim_10: int = 4, max_dim: int = 512) -> int:
    """Official ``nd(size)``: 512 at 4px/8px, halving per doubling of resolution."""
    return min(int(2 ** (10 - math.log2(size)) * dim_10), max_dim)


def official_init_(module: nn.Module) -> None:
    """``tl.get_initializer(leaky_relu)``: variance scaling, fan_in, truncated normal.

    ``tl.get_initializer`` is ``variance_scaling(scale=gain**2, mode='fan_in',
    distribution='truncated_normal')`` (tflib/deep_learning/utils.py:82-84), and
    ``weight[0].numel()`` equals TF's ``fan_in`` for all three layer types. TF/tflib
    kernels are ``[in, out]`` / ``[kh, kw, in, out]`` / ``[kh, kw, out, in]``
    (tflib/deep_learning/layers.py:29/99/168) against PyTorch's ``[out, in]`` /
    ``[out, in, kh, kw]`` / ``[in, out, kh, kw]``:

    * Linear: 2-D shapes take ``fan_in = shape[0]`` (init_ops.py:1428-1430).
    * Conv2d: deeper shapes take the ``_compute_fans`` else branch
      (init_ops.py:1431-1438), verbatim::

          # Assuming convolution kernels (2D, 3D, or more).
          # kernel shape: (..., input_depth, depth)
          receptive_field_size = 1
          for dim in shape[:-2]:
            receptive_field_size *= dim
          fan_in = shape[-2] * receptive_field_size   # :1437

    * ConvTranspose2d: ``_compute_fans`` never special-cases transposed filters,
      and the official ``conv2d_transpose`` filter is ``[height, width,
      output_channels, in_channels]`` (nn_ops.py:2160-2162) — so ``shape[-2]`` is
      the *output* channels despite the "input_depth" comment, and TF's ``fan_in``
      is ``out * kh * kw``, again ``weight[0].numel()``.
    """
    gain = math.sqrt(2.0 / (1.0 + LEAKY_SLOPE**2))
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            std = gain / math.sqrt(layer.weight[0].numel()) / _TRUNC_STD
            nn.init.trunc_normal_(layer.weight, 0.0, std, -2.0 * std, 2.0 * std)
            nn.init.zeros_(layer.bias)


def orthogonal_regularizer(basis: torch.Tensor) -> torch.Tensor:
    """Official ``0.5 * sum((UᵀU - I)²)`` for one layer; ``basis`` is ``[q, C, H, W]``."""
    flat = basis.flatten(1)
    gram = flat @ flat.t()
    eye = torch.eye(flat.size(0), device=flat.device, dtype=flat.dtype)
    return 0.5 * (gram - eye).square().sum()


def hinge_v1_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_v1_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    # max(1 - f, 0), not -mean(f): the two differ once a fake logit exceeds 1.
    return F.relu(1.0 - fake_logits).mean()


def zero_centered_gradient_penalty(real_logits: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    """Official ``0-gp`` on real samples: ``mean(||∇ₓ D(x)||²)`` — no 0.5 factor."""
    (grad,) = torch.autograd.grad(real_logits.sum(), real_images, create_graph=True)
    return grad.flatten(1).square().sum(dim=1).mean()


def _resolution_levels(image_size: int) -> int:
    levels = round(math.log2(image_size / 4)) if image_size >= 4 else -1
    if levels < 1 or 4 * 2**levels != image_size:
        raise ValueError(f"image_size must be 4 * 2**n with n >= 1, got {image_size}")
    return levels


class EigenGANGenerator(nn.Module):
    """One linear subspace ``(U_i, L_i, mu_i)`` per layer, each fed by its own ``z_i``.

    Layer ``i`` works at ``4 * 2**i`` px: ``phi_i = mu_i + sum_j (L_ij * z_ij) * U_ij`` has
    the feature-map shape ``[C, H, W]`` and is added to the features before *both* the
    stride-2 and the stride-1 transposed convolution of that layer.

    ``use_gram_schmidt`` is the official ``--g_loss_weight_orth_loss 0`` mode: the forward
    pass orthonormalizes ``U`` with Gram-Schmidt instead of relying on the penalty.
    """

    def __init__(
        self,
        z_dims: tuple[int, ...] | list[int] = (6,) * 6,
        eps_dim: int = 512,
        n_channels: int = 3,
        dim_10: int = 4,
        use_gram_schmidt: bool = False,
    ):
        super().__init__()
        self.z_dims = tuple(int(q) for q in z_dims)
        if not self.z_dims or min(self.z_dims) < 1:
            raise ValueError("z_dims must be a non-empty sequence of positive ints")
        self.eps_dim = int(eps_dim)
        self.use_gram_schmidt = bool(use_gram_schmidt)
        self.image_size = 4 * 2 ** len(self.z_dims)
        nd = lambda size: channels(size, dim_10)  # noqa: E731

        self.fc = nn.Linear(self.eps_dim, 4 * 4 * nd(4))
        self.U = nn.ParameterList()
        self.L = nn.ParameterList()
        self.mu = nn.ParameterList()
        self.inject_pre = nn.ModuleList()
        self.inject_post = nn.ModuleList()
        self.up = nn.ModuleList()
        self.refine = nn.ModuleList()
        for i, q in enumerate(self.z_dims):
            size = 4 * 2**i
            c_in, c_out = nd(size), nd(size * 2)
            basis = torch.empty(q, c_in, size, size)
            nn.init.orthogonal_(basis)  # orthonormal rows of the flattened [q, C*H*W]
            self.U.append(nn.Parameter(basis))
            self.L.append(nn.Parameter(torch.tensor([3.0 * k for k in range(q, 0, -1)])))
            self.mu.append(nn.Parameter(torch.zeros(c_in, size, size)))
            self.inject_pre.append(nn.ConvTranspose2d(c_in, c_in, 1, 1))
            self.inject_post.append(SameStride2ConvTranspose2d(c_in, c_out))
            self.up.append(SameStride2ConvTranspose2d(c_in, c_out))
            self.refine.append(nn.ConvTranspose2d(c_out, c_out, 3, 1, padding=1))
        self.to_rgb = nn.Conv2d(nd(self.image_size), n_channels, 7, 1, padding=3)
        # fc/conv layers only; U / L / mu keep the orthogonal / constant / zeros init above.
        official_init_(self)

    def _draw(self, shape: tuple[int, ...], truncation: float | None) -> torch.Tensor:
        x = torch.empty(shape, device=self.fc.weight.device)
        if truncation is None:
            return x.normal_()
        return nn.init.trunc_normal_(x, 0.0, 1.0, -truncation, truncation)

    def sample_latents(self, batch: int, truncation: float | None = None):
        """Training uses plain N(0, 1); sampling uses N(0, 1) truncated to ±``truncation``.

        ``nn.init.trunc_normal_`` is the same inverse-CDF truncated normal as the official
        ``tl.truncated_normal`` (``tflib/random/random.py:4``), not a clipped normal.
        """
        zs = [self._draw((batch, q), truncation) for q in self.z_dims]
        return zs, self._draw((batch, self.eps_dim), truncation)

    def dims_by_importance(self, layer: int) -> torch.Tensor:
        """Official traversal order: dimensions sorted by descending ``|L|``."""
        return torch.argsort(self.L[layer].detach().abs(), descending=True)

    def orthogonal_loss(self) -> torch.Tensor:
        """Sum over layers of the official regularizer (weight 1 in the official trainer)."""
        return torch.stack([orthogonal_regularizer(basis) for basis in self.U]).sum()

    def _basis(self, layer: int) -> torch.Tensor:
        basis = self.U[layer]
        if self.use_gram_schmidt:
            return gram_schmidt(basis.flatten(1)).view_as(basis)
        return basis

    def forward(self, zs: list[torch.Tensor], eps: torch.Tensor) -> torch.Tensor:
        if len(zs) != len(self.z_dims):
            raise ValueError(f"expected {len(self.z_dims)} latent tensors, got {len(zs)}")
        if eps.ndim != 2 or eps.shape[1] != self.eps_dim:
            raise ValueError(f"eps must have shape [batch, {self.eps_dim}]")
        h = self.fc(eps).view(eps.size(0), -1, 4, 4)
        for i, z in enumerate(zs):
            if z.ndim != 2 or z.shape[1] != self.z_dims[i]:
                raise ValueError(f"zs[{i}] must have shape [batch, {self.z_dims[i]}]")
            phi = torch.einsum("bq,qchw->bchw", self.L[i] * z, self._basis(i)) + self.mu[i]
            h = self.up[i](F.leaky_relu(h + self.inject_pre[i](phi), LEAKY_SLOPE))
            h = self.refine[i](F.leaky_relu(h + self.inject_post[i](phi), LEAKY_SLOPE))
        return torch.tanh(self.to_rgb(F.leaky_relu(h, LEAKY_SLOPE)))

    def traverse(
        self,
        zs: list[torch.Tensor],
        eps: torch.Tensor,
        layer: int,
        dim: int,
        values: list[float],
    ) -> Iterator[torch.Tensor]:
        """One image batch per value of ``zs[layer][:, dim]`` (official ``traversal_graph``).

        Yields lazily so a caller can move each frame off the GPU before the next one.
        """
        for value in values:
            edited = [z.clone() for z in zs]
            edited[layer][:, dim] = value
            with torch.no_grad():
                frame = self(edited, eps)
            yield frame


class EigenGANDiscriminator(nn.Module):
    """Official D: 7x7 conv, then per level (3x3 s1, 3x3 s2), 3x3 s1, fc(512), fc(1).

    No spectral norm and no feature normalization (the official defaults).
    """

    def __init__(
        self,
        image_size: int = 256,
        n_channels: int = 3,
        dim_10: int = 4,
        fc_dim: int = 1024,
        max_dim: int = 512,
    ):
        super().__init__()
        levels = _resolution_levels(image_size)
        nd = lambda size: channels(size, dim_10, max_dim)  # noqa: E731
        act = lambda: nn.LeakyReLU(LEAKY_SLOPE)  # noqa: E731

        size = image_size
        layers: list[nn.Module] = [nn.Conv2d(n_channels, nd(size), 7, 1, padding=3), act()]
        for _ in range(levels):
            layers += [nn.Conv2d(nd(size), nd(size), 3, 1, padding=1), act()]
            layers += [SameStride2Conv2d(nd(size), nd(size // 2)), act()]
            size //= 2
        layers += [nn.Conv2d(nd(size), nd(size), 3, 1, padding=1), act()]
        self.features = nn.Sequential(*layers)
        hidden = min(fc_dim, max_dim)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(4 * 4 * nd(4), hidden), act(), nn.Linear(hidden, 1)
        )
        official_init_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
