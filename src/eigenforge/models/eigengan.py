import math

import torch
import torch.nn.functional as F
from torch import nn


class SubspaceLayer(nn.Module):
    """Layer-wise orthogonal latent subspace used by EigenGAN.

    ``basis`` stores K row-wise directions.  The forward path can use a QR
    projection for inference while ``orthogonality_loss`` keeps the learnable
    parameters close to the Stiefel manifold during training.
    """

    def __init__(
        self,
        in_dim: int | None = None,
        out_dim: int | None = None,
        n_basis: int = 6,
        *,
        dim: int | None = None,
        orthonormal_init: bool = True,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        if out_dim is None:
            raise ValueError("out_dim or dim must be provided")
        if in_dim is None:
            in_dim = out_dim

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_basis = int(n_basis)
        if self.n_basis <= 0:
            raise ValueError("n_basis must be positive")
        if self.out_dim <= 0:
            raise ValueError("out_dim must be positive")
        if self.n_basis > self.out_dim:
            raise ValueError(
                f"n_basis ({self.n_basis}) cannot exceed out_dim ({self.out_dim}) for subspace projection"
            )
        self.basis = nn.Parameter(torch.empty(self.n_basis, self.out_dim))
        if orthonormal_init:
            nn.init.orthogonal_(self.basis)
        else:
            nn.init.kaiming_normal_(self.basis)
        self.fc_alpha = (
            nn.Identity()
            if self.in_dim == self.n_basis
            else nn.Linear(self.in_dim, self.n_basis)
        )
        # EigenGAN's diagonal L_i importance keeps a stable total style scale.
        self.log_importance = nn.Parameter(torch.zeros(self.n_basis))
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def get_orthonormal_basis(self) -> torch.Tensor:
        q, r = torch.linalg.qr(self.basis.t(), mode="reduced")
        # Householder QR leaves the sign of each column free, so a column of Q flips
        # (cosine -1) the instant a basis entry crosses zero, which flips a whole
        # semantic axis mid-training. Forcing diag(R) >= 0 makes the factorization
        # unique and Q continuous in the basis. The sign is piecewise constant, so
        # it does not change gradients; the orthogonality loss never sees Q.
        sign = torch.where(torch.diagonal(r) < 0, -1.0, 1.0).to(q.dtype)
        return (q * sign).t()

    def orthogonality_loss(self, scale: str = "normalized") -> torch.Tensor:
        """Return Gram-matrix orthogonality penalty.

        Args:
            scale: "normalized" (default) uses F.mse_loss (Frobenius^2 / n_basis^2),
                   preserving the project notebook's baseline training dynamics.
                   "sum_half" uses official 0.5 * sum((Gram - I)^2) per LynnHo/EigenGAN.
        """
        gram = self.basis @ self.basis.t()
        identity = torch.eye(self.n_basis, device=self.basis.device, dtype=self.basis.dtype)
        if scale == "sum_half":
            return 0.5 * (gram - identity).square().sum()
        if scale == "normalized":
            return F.mse_loss(gram, identity)
        raise ValueError(f"Unknown scale: {scale!r}, expected 'normalized' or 'sum_half'")

    # Explicit name used by the original notebook's terminology.
    get_orthogonality_loss = orthogonality_loss

    def importance(self) -> torch.Tensor:
        """Return positive diagonal L_i weights with mean scale equal to one."""
        return self.log_importance.softmax(dim=0) * self.n_basis

    def forward(
        self,
        z: torch.Tensor,
        alpha_override: torch.Tensor | None = None,
        strict_qr: bool = True,
    ) -> torch.Tensor:
        alpha = self.fc_alpha(z) if alpha_override is None else alpha_override
        if alpha.ndim != 2 or alpha.shape[1] != self.n_basis:
            raise ValueError(f"alpha must have shape [batch, {self.n_basis}]")
        basis = self.get_orthonormal_basis() if strict_qr else self.basis
        basis = basis * self.importance().unsqueeze(1)
        return alpha @ basis + self.bias


class ResBlockUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        # NOT in-place: the skip conv reuses this activation's input for its
        # weight gradient, so an in-place LeakyReLU corrupts backward.
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, scale_factor=2, mode="nearest")
        residual = self.skip(h)
        h = self.conv1(self.act(h))
        h = self.conv2(self.act(h))
        return h + residual


class EigenGenerator(nn.Module):
    """A compact, layer-wise semantically controllable face generator."""

    def __init__(
        self,
        img_size: int = 64,
        # Aligned with the training/traversal CLI defaults (--z-dim 128): a
        # latent_dim default that disagrees with the entry points only shows up
        # as an opaque load_state_dict shape error.
        latent_dim: int = 128,
        n_basis: int = 6,
        base_channels: int = 64,
        max_channels: int = 512,
        *,
        z_dim: int | None = None,
        img_channels: int = 3,
    ):
        super().__init__()
        if z_dim is not None:
            latent_dim = z_dim
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if max_channels < base_channels:
            raise ValueError("max_channels cannot be smaller than base_channels")
        if img_size < 8 or img_size & (img_size - 1):
            raise ValueError("img_size must be a power of two and at least 8")

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.z_dim = latent_dim
        self.n_basis = n_basis
        self.n_stages = int(math.log2(img_size)) - 1

        def channels(stage: int) -> int:
            return min(max_channels, base_channels * (2 ** (self.n_stages - 1 - stage)))

        self.initial_channels = channels(0)
        self.initial_const = nn.Parameter(torch.randn(1, self.initial_channels, 4, 4))
        self.subspace_layers = nn.ModuleList(
            [SubspaceLayer(latent_dim, channels(i), n_basis) for i in range(self.n_stages)]
        )

        self.blocks = nn.ModuleList()
        in_channels = self.initial_channels
        for i in range(self.n_stages - 1):
            out_channels = channels(i + 1)
            self.blocks.append(ResBlockUp(in_channels, out_channels))
            in_channels = out_channels

        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, img_channels, 3, padding=1),
            nn.Tanh(),
        )

    def sample_latents(self, batch: int, truncation: float = 1.0):
        device = self.initial_const.device
        z = torch.randn(batch, self.latent_dim, device=device) * truncation
        return z

    def forward(
        self,
        z: torch.Tensor,
        layer_edits: dict[int, torch.Tensor] | None = None,
        strict_qr: bool = True,
    ) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape [batch, {self.latent_dim}]")
        h = self.initial_const.expand(z.size(0), -1, -1, -1)
        for i, block in enumerate(self.blocks):
            alpha = layer_edits.get(i) if layer_edits else None
            style = self.subspace_layers[i](z, alpha_override=alpha, strict_qr=strict_qr)
            if style.shape[1] != h.shape[1]:
                raise RuntimeError("subspace and feature channels are inconsistent")
            h = block(h + style[:, :, None, None])

        final_idx = self.n_stages - 1
        alpha = layer_edits.get(final_idx) if layer_edits else None
        style = self.subspace_layers[final_idx](z, alpha_override=alpha, strict_qr=strict_qr)
        h = h + style[:, :, None, None]
        return self.to_rgb(h)

    def sample_with_eigen_control(
        self, z: torch.Tensor, layer_idx: int, basis_idx: int, alpha: float
    ) -> torch.Tensor:
        if not 0 <= layer_idx < self.n_stages:
            raise IndexError("layer_idx out of range")
        if not 0 <= basis_idx < self.n_basis:
            raise IndexError("basis_idx out of range")
        coords = torch.zeros(z.size(0), self.n_basis, device=z.device, dtype=z.dtype)
        coords[:, basis_idx] = alpha
        return self.forward(z, layer_edits={layer_idx: coords}, strict_qr=True)

    def get_total_orthogonality_loss(self) -> torch.Tensor:
        return torch.stack([layer.orthogonality_loss() for layer in self.subspace_layers]).mean()

    total_orthogonality_loss = get_total_orthogonality_loss


class EigenDiscriminator(nn.Module):
    """Spectral-normalized hinge-loss discriminator for 64x64 RGB faces."""

    def __init__(self, img_size: int = 64, img_channels: int = 3, base_channels: int = 64):
        super().__init__()
        layers = []
        in_channels = img_channels
        size = img_size
        while size > 4:
            out_channels = min(base_channels * (2 ** len(layers)), 512)
            layers.extend(
                [
                    nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 2, 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = out_channels
            size //= 2
        self.features = nn.Sequential(*layers)
        self.head = nn.utils.spectral_norm(nn.Conv2d(in_channels, 1, 4, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).flatten(1)


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def r1_penalty(real_logits: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    """Compute the half-scaled R1 gradient penalty for real samples."""
    gradients = torch.autograd.grad(
        outputs=real_logits,
        inputs=real_images,
        grad_outputs=torch.ones_like(real_logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return 0.5 * gradients.square().flatten(1).sum(dim=1).mean()
