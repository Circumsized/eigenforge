"""Linear-case EigenGAN under maximum likelihood (paper Sec 3.2 + Appendix A).

Closed-form MLE of x = U L z + mu + sigma*eps with orthonormal U [d, q], diagonal
L >= 0 and z ~ N_q(0, I) — Eq. (43)-(46) of arXiv:2104.12476v2, Appendix A:

    mu = mean(x)                                (43)
    sigma^2 = (tr(S) - tr(Lambda_q)) / (d - q)  (44)  Lambda_q = top-q eigenvalues
    L^2 = Lambda_q - sigma^2 I                  (45)
    U = the top-q eigenvectors of S             (46)

with S the data covariance under the paper's 1/n normalization, Eq. (17). The basis
therefore equals PCA's — the paper's theoretical core (Sec 3.2). Unlike the Tables 2-3
adversarial toy experiment (whose optimizer, iterations, discriminator and A-rank
construction the paper never specifies), every input here is pinned by the text, so
this module involves no guessed hyperparameters.

Never executed. ``samples`` are the n observations; ``subspace_dim`` is q, required
q < d because Eq. (44) divides by (d - q).
"""
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearCaseMLE:
    """MLE parameters of the one-layer EigenGAN, Eq. (43)-(46)."""

    mu: torch.Tensor  # [d]     Eq. (43)
    sigma_sq: torch.Tensor  # scalar, >= 0  Eq. (44)
    basis: torch.Tensor  # [d, q]  orthonormal columns, top-q eigenvectors, Eq. (46)
    importance: torch.Tensor  # [q]  L, non-negative, descending, Eq. (45)


def linear_case_mle(samples: torch.Tensor, subspace_dim: int) -> LinearCaseMLE:
    """Fit the closed-form MLE; deterministic given ``samples``.

    Numerical notes: the fit runs entirely in float64 regardless of the input
    dtype and the results are cast back to it; eigenvalues just below sigma^2
    (possible when lambda_q ~= sigma^2, e.g. under eigenvalue ties) give L^2
    slightly negative, clamped to 0; sigma^2 is a mean of non-negative
    eigenvalues and is clamped the same way against roundoff.
    """
    if (
        not isinstance(samples, torch.Tensor)
        or not samples.is_floating_point()
        or samples.ndim != 2
        or samples.numel() == 0
    ):
        # Integer samples would pass isfinite and the final .to(dtype=...) would
        # truncate the orthonormal basis to zeros — refuse instead.
        raise ValueError("samples must be a non-empty floating-point [n, d] tensor")
    n, d = samples.shape
    if isinstance(subspace_dim, bool) or not isinstance(subspace_dim, int):
        raise ValueError("subspace_dim must be an int")
    if not 1 <= subspace_dim < d:
        raise ValueError(f"subspace_dim must satisfy 1 <= q < d (got q={subspace_dim}, d={d})")
    if not torch.isfinite(samples).all():
        raise ValueError("samples contain non-finite values")
    if n < 2:
        raise ValueError("need at least 2 observations to form a covariance")

    # Everything runs in float64: the closed form's selling point is exactness, and the
    # noise end of the eigenvalue spectrum is where float32 accumulation shows first.
    # Results are cast back to the input dtype at the end.
    samples64 = samples.to(dtype=torch.float64)
    mu = samples64.mean(dim=0)  # Eq. (43)
    centered = samples64 - mu
    # Eq. (17) normalizes by n (not n-1); eigenvectors are unaffected, L/sigma scale.
    covariance = centered.t() @ centered / n
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = torch.flip(eigenvalues, dims=(0,))  # descending, as Eq. (46) orders
    basis = torch.flip(eigenvectors, dims=(1,))[:, :subspace_dim]

    sigma_sq = (eigenvalues.sum() - eigenvalues[:subspace_dim].sum()) / (d - subspace_dim)
    if not math.isfinite(sigma_sq.item()) or sigma_sq.item() < 0:
        if sigma_sq.item() < 0 and sigma_sq.item() > -1e-8:
            sigma_sq = sigma_sq.clamp_min(0.0)  # roundoff guard; Eq. (44) is >= 0
        else:
            raise ValueError(f"sigma^2 is not a valid non-negative value: {sigma_sq.item()}")
    importance = (eigenvalues[:subspace_dim] - sigma_sq).clamp_min(0.0).sqrt()  # Eq. (45)

    mu_out = mu.to(dtype=samples.dtype)
    return LinearCaseMLE(
        mu=mu_out,
        sigma_sq=sigma_sq.to(dtype=samples.dtype),
        basis=basis.to(dtype=samples.dtype),
        importance=importance.to(dtype=samples.dtype),
    )
