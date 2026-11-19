"""Tables 2-3 of EigenGAN (arXiv:2104.12476v2, Sec 4.2): linear EigenGAN vs PCA basis.

Paper-specified, implemented verbatim: toy data y = A x + b (Eq. 11) with
x ~ N_d(0, I) or U_d(0, 1); the linear model x_hat = U L z + mu + sigma*eps
(Eq. 5; U with orthonormal rows, L diagonal); adversarial training with
Vanilla GAN / LSGAN / WGAN / HingeGAN / KL-f-GAN (formulas verbatim from the
official ``tfprob/gan/loss.py``; hinge reuses the verified ``hinge_v1`` pair);
cosine similarity between the learned basis and the PCA basis, averaged over
100 random toy datasets; the PCA basis comes from ``linear_case_mle`` (whose
closed form is exactly PCA, Appendix A Eq. 43-46).

Never executed. The paper does NOT give the training protocol, so every
choice marked (*) below is an explicit default, not a reproduction claim —
the output is therefore NOT comparable to the paper's Table 2/3 numbers, only
qualitatively to their conclusion (an adversarially trained linear EigenGAN
basis should approach the PCA basis).

Paper-silent defaults (*): "data rank" == ambient dimension d, A ~ N(0,I) full
rank, b ~ N(0,I_d); discriminator is one affine layer (nn.Linear(d, 1));
Adam(lr 1e-4, betas (0.5, 0.999)) mirroring the official general-case
defaults; 2000 iterations, batch 128, 10000 samples per dataset; no
regularization except the orthogonality term at the official weight 1;
similarity = mean_j |cos(u_j, v_j)| with learned rows sorted by descending
|L| and PCA columns by descending eigenvalue (signs of eigenvectors are
arbitrary, hence the absolute value); "HingeGAN" = hinge_v1 (the official CLI
default); KL-f-GAN uses the tricky generator loss (official default).
Full default sweep is hours of CPU — scale down via the flags.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eigenforge.common.logger import logger
from eigenforge.evaluation.linear_case_mle import linear_case_mle
from eigenforge.models.eigengan_faithful import (
    hinge_v1_d_loss,
    hinge_v1_g_loss,
    orthogonal_regularizer,
)

LOSSES = ("gan", "lsgan", "wgan", "hinge_v1", "fgan_kl")


def get_adversarial_losses(mode: str):
    """PyTorch mirror of official ``tfprob/gan/loss.py::get_adversarial_losses_fn``."""
    if mode == "hinge_v1":
        return hinge_v1_d_loss, hinge_v1_g_loss
    if mode == "gan":  # official: BinaryCrossentropy(from_logits=True)
        def d_loss(r_logit, f_logit):
            return (
                F.binary_cross_entropy_with_logits(r_logit, torch.ones_like(r_logit))
                + F.binary_cross_entropy_with_logits(f_logit, torch.zeros_like(f_logit))
            )

        def g_loss(f_logit):
            return F.binary_cross_entropy_with_logits(f_logit, torch.ones_like(f_logit))

        return d_loss, g_loss
    if mode == "lsgan":  # official: MeanSquaredError against 1/0 labels
        def d_loss(r_logit, f_logit):
            return (
                F.mse_loss(r_logit, torch.ones_like(r_logit))
                + F.mse_loss(f_logit, torch.zeros_like(f_logit))
            )

        def g_loss(f_logit):
            return F.mse_loss(f_logit, torch.ones_like(f_logit))

        return d_loss, g_loss
    if mode == "wgan":  # official: d = -mean(r) + mean(f); g = -mean(f)
        def d_loss(r_logit, f_logit):
            return -r_logit.mean() + f_logit.mean()

        def g_loss(f_logit):
            return -f_logit.mean()

        return d_loss, g_loss
    if mode == "fgan_kl":
        # official get_divergence_fn('KL'): T(v)=v, T*(t)=exp(t-1); tricky g = -mean(T(f))
        def d_loss(r_logit, f_logit):
            return -r_logit.mean() + (f_logit - 1).exp().mean()

        def g_loss(f_logit):
            return -f_logit.mean()

        return d_loss, g_loss
    raise ValueError(f"unknown adversarial loss mode: {mode!r}")


class LinearEigenGAN(nn.Module):
    """Eq. (5): x_hat = U L z + mu + sigma*eps, one layer, vector data."""

    def __init__(self, data_dim: int, subspace_dim: int):
        super().__init__()
        if not 1 <= subspace_dim < data_dim:
            raise ValueError(f"need 1 <= q < d, got q={subspace_dim}, d={data_dim}")
        self.data_dim = data_dim
        self.subspace_dim = subspace_dim
        self.U = nn.Parameter(torch.empty(subspace_dim, data_dim))
        nn.init.orthogonal_(self.U)  # rows orthonormal (q < d)
        # (*): L init [3q..3] — the official general-case convention.
        self.L = nn.Parameter(torch.tensor([3.0 * k for k in range(subspace_dim, 0, -1)]))
        self.mu = nn.Parameter(torch.zeros(data_dim))
        self.log_sigma = nn.Parameter(torch.zeros(()))  # (*): sigma = 1 at init

    def forward(self, batch: int) -> torch.Tensor:
        z = torch.randn(batch, self.subspace_dim, device=self.U.device)
        eps = torch.randn(batch, self.data_dim, device=self.U.device)
        return (z * self.L) @ self.U + self.mu + self.log_sigma.exp() * eps

    def basis_by_importance(self) -> torch.Tensor:
        """Rows sorted by descending |L| to pair with PCA's eigenvalue order."""
        order = torch.argsort(self.L.detach().abs(), descending=True)
        return self.U.detach()[order]


def make_toy_dataset(n_samples: int, data_dim: int, distribution: str) -> torch.Tensor:
    """Eq. (11): y = A x + b. (*): A ~ N(0,I) full rank, b ~ N(0,I_d)."""
    if distribution == "gaussian":
        x = torch.randn(n_samples, data_dim)
    elif distribution == "uniform":
        x = torch.rand(n_samples, data_dim)  # U_d(0, 1)
    else:
        raise ValueError(f"distribution must be 'gaussian' or 'uniform', got {distribution!r}")
    transform = torch.randn(data_dim, data_dim)
    translation = torch.randn(data_dim)
    return x @ transform.t() + translation


def train_toy_basis(
    data: torch.Tensor,
    subspace_dim: int,
    mode: str,
    *,
    iters: int,
    batch_size: int,
    lr: float,
) -> torch.Tensor:
    """Adversarially train one linear EigenGAN; return its importance-sorted basis."""
    generator = LinearEigenGAN(data.size(1), subspace_dim)
    discriminator = nn.Linear(data.size(1), 1)  # (*): single affine layer
    d_loss_fn, g_loss_fn = get_adversarial_losses(mode)
    g_optim = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    n = data.size(0)
    for _ in range(iters):
        real = data[torch.randint(0, n, (batch_size,))]
        fake = generator(batch_size)
        d_loss = d_loss_fn(discriminator(real), discriminator(fake.detach()))
        d_optim.zero_grad(set_to_none=True)
        d_loss.backward()
        d_optim.step()

        # Fresh fake for the G step, as the official trainer does.
        g_loss = g_loss_fn(discriminator(generator(batch_size))) + orthogonal_regularizer(
            generator.U
        )  # orthogonality at the official weight 1; no other regularization (*)
        g_optim.zero_grad(set_to_none=True)
        g_loss.backward()
        g_optim.step()
    return generator.basis_by_importance()


def basis_similarity(learned_rows: torch.Tensor, pca_columns: torch.Tensor) -> float:
    """mean_j |cos(u_j, v_j)|; (*): absolute value — eigenvector signs are arbitrary."""
    learned = F.normalize(learned_rows.float(), dim=1)
    pca = F.normalize(pca_columns.t().float(), dim=1)  # [q, d], eigenvalue-descending
    return (learned * pca).sum(dim=1).abs().mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="EigenGAN Tables 2-3: basis similarity with PCA")
    parser.add_argument(
        "--configs", default="5:1,5:3,10:1,10:3,10:5,20:1,20:5,20:10",
        help="data_rank:subspace_rank pairs, the paper's table header",
    )
    parser.add_argument(
        "--losses",
        default=",".join(LOSSES),
        help=f"comma-separated non-empty subset of {list(LOSSES)} (default: all five)",
    )
    parser.add_argument("--distribution", default="both", choices=["gaussian", "uniform", "both"])
    parser.add_argument("--n-datasets", type=int, default=100, help="paper: average over 100 datasets")
    parser.add_argument("--n-samples", type=int, default=10000, help="(*): paper-silent")
    parser.add_argument("--iters", type=int, default=2000, help="(*): paper-silent")
    parser.add_argument("--batch-size", type=int, default=128, help="(*): paper-silent")
    parser.add_argument("--lr", type=float, default=1e-4, help="(*): official general-case default")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.n_datasets, args.n_samples, args.iters, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("count arguments must be >= 1 and --lr > 0")
    configs = []
    for item in args.configs.split(","):
        rank, subspace = (int(value) for value in item.split(":"))
        if not 1 <= subspace < rank:
            raise ValueError(f"config {item!r} must satisfy 1 <= subspace_rank < data_rank")
        configs.append((rank, subspace))
    # argparse choices cannot express "any non-empty subset of LOSSES" (a single
    # choices entry would only accept the full default string), so the split is
    # validated here — an unknown loss name must fail loudly, not silently train
    # the default set.
    losses = [name.strip() for name in args.losses.split(",")]
    unknown = [name for name in losses if name not in LOSSES]
    if not losses or unknown:
        raise ValueError(
            f"--losses must be a non-empty subset of {list(LOSSES)}, got {args.losses!r}"
            + (f"; unknown: {unknown}" if unknown else "")
        )
    distributions = ["gaussian", "uniform"] if args.distribution == "both" else [args.distribution]
    torch.manual_seed(args.seed)

    for distribution in distributions:
        totals = {mode: {config: 0.0 for config in configs} for mode in losses}
        for dataset_index in range(args.n_datasets):
            for config in configs:
                data = make_toy_dataset(args.n_samples, config[0], distribution)
                pca_basis = linear_case_mle(data, config[1]).basis
                for mode in losses:
                    learned = train_toy_basis(
                        data, config[1], mode,
                        iters=args.iters, batch_size=args.batch_size, lr=args.lr,
                    )
                    totals[mode][config] += basis_similarity(learned, pca_basis)
            if (dataset_index + 1) % 10 == 0:
                logger.info("{}: {}/{} datasets done", distribution, dataset_index + 1, args.n_datasets)
        header = "loss".ljust(10) + "".join(f"{d}->{q}".rjust(9) for d, q in configs)
        print(f"\nP = {'N_d(0, I)' if distribution == 'gaussian' else 'U_d(0, 1)'}")
        print(header)
        for mode in losses:
            row = mode.ljust(10)
            for config in configs:
                row += f"{totals[mode][config] / args.n_datasets:9.3f}"
            print(row)


if __name__ == "__main__":
    main()
