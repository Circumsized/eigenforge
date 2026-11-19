"""EigenGAN Table 1 entropy coefficient, Eq. (6)-(10) (arXiv:2104.12476v2).

The paper discretizes U[-4.5, 4.5] into 100 equal bins with 1000 images per bin.
U(Y|Z) = (H(Y) - H(Y|Z)) / H(Y) measures the fraction of attribute information in Z.
Reproduction target (paper Table 1, "Ours" diagonal): Gender 57%, Eyeglasses 33%,
Smiling 55%, Black Hair 38%; the full matrix also carries the off-diagonal entanglement
terms that the table reports per method.

Requires a caller-supplied CelebA attribute classifier. The paper uses 40 binary
attributes but does not specify the classifier architecture or its training.
Neither the estimator nor its sampling code has been executed in this project.

Explicit estimator choices not fully specified in the paper: z is uniform inside
its bin; other latents are independent N(0, 1); generator images are in [-1, 1].
The classifier callable handles any required preprocessing and returns probabilities
[samples, attributes], not logits or hard labels. Modules must already be in eval mode.
"""
import math
from collections.abc import Callable

import torch


def _validate_probabilities(p: torch.Tensor) -> None:
    if not isinstance(p, torch.Tensor) or not p.is_floating_point() or p.numel() == 0:
        raise ValueError("probabilities must be a non-empty floating-point tensor")
    if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]; apply sigmoid to logits first")


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Entropy in nats, computed in float64; xlogy defines 0 * log(0) as 0."""
    _validate_probabilities(p)
    p = p.to(dtype=torch.float64)
    return -(torch.special.xlogy(p, p) + torch.special.xlogy(1.0 - p, 1.0 - p))


def entropy_coefficient(p_y_given_z: torch.Tensor) -> torch.Tensor:
    """U(Y|Z) from a [bins, attributes] probability grid, one coefficient per attribute.

    Equal-width bins under a uniform p_Z have equal weights. An attribute with
    marginal probability exactly 0 or 1 has H(Y)=0 and returns NaN (undefined).
    A constant probability strictly between 0 and 1 has H(Y)>0 and coefficient 0.
    """
    _validate_probabilities(p_y_given_z)
    if p_y_given_z.ndim != 2 or p_y_given_z.shape[0] < 2 or p_y_given_z.shape[1] < 1:
        raise ValueError("p_y_given_z must have shape [n_bins >= 2, n_attrs >= 1]")
    p = p_y_given_z.to(dtype=torch.float64)
    h_y_given_z = binary_entropy(p).mean(dim=0)
    h_y = binary_entropy(p.mean(dim=0))
    defined = h_y > 0
    denominator = torch.where(defined, h_y, torch.ones_like(h_y))
    # Concavity implies [0, 1]; clamp only the floating-point roundoff at the bounds.
    coefficient = ((h_y - h_y_given_z) / denominator).clamp(0.0, 1.0)
    return torch.where(defined, coefficient, torch.full_like(coefficient, float("nan")))


@torch.no_grad()
def attribute_posterior_along_dim(
    generator,
    classifier: Callable[[torch.Tensor], torch.Tensor],
    layer: int,
    dim: int,
    *,
    n_bins: int = 100,
    n_per_bin: int = 1000,
    z_range: float = 4.5,
    batch_size: int = 250,
) -> torch.Tensor:
    """Estimate p(y=1 | z_dim in bin), returned on CPU as float64 [bins, attributes].

    Override only zs[layer][:, dim]. Sum classifier probabilities batch by batch,
    avoiding retention of images or all predictions for the 1000 samples in a bin.
    The caller owns device placement, preprocessing, RNG seed and eval mode.
    """
    for name, value, minimum in (("n_bins", n_bins, 2), ("n_per_bin", n_per_bin, 1),
                                  ("batch_size", batch_size, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not math.isfinite(z_range) or z_range <= 0:
        raise ValueError("z_range must be finite and positive")
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < len(generator.z_dims):
        raise IndexError("layer is out of range")
    if not isinstance(dim, int) or isinstance(dim, bool) or not 0 <= dim < generator.z_dims[layer]:
        raise IndexError("dim is out of range")
    if generator.training or (isinstance(classifier, torch.nn.Module) and classifier.training):
        raise ValueError("generator and classifier modules must be in eval mode")

    edges = torch.linspace(-z_range, z_range, n_bins + 1, dtype=torch.float64)
    rows = []
    n_attrs = None
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
        posterior_sum = None
        for start in range(0, n_per_bin, batch_size):
            n = min(batch_size, n_per_bin - start)
            zs, eps = generator.sample_latents(n)
            zs[layer][:, dim] = lo + (hi - lo) * torch.rand(n, device=eps.device, dtype=eps.dtype)
            probabilities = classifier(generator(zs, eps))
            _validate_probabilities(probabilities)
            if probabilities.ndim != 2 or probabilities.shape[0] != n or probabilities.shape[1] < 1:
                raise ValueError("classifier must return [batch, n_attrs >= 1] probabilities")
            if n_attrs is None:
                n_attrs = probabilities.shape[1]
            elif probabilities.shape[1] != n_attrs:
                raise ValueError("classifier attribute count changed between batches")
            chunk_sum = probabilities.sum(dim=0, dtype=torch.float64).cpu()
            posterior_sum = chunk_sum if posterior_sum is None else posterior_sum + chunk_sum
        rows.append(posterior_sum / n_per_bin)
    return torch.stack(rows)
