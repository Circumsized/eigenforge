import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from eigenforge.common.logger import logger


class GenAIEvaluator:
    """Evaluation helpers with optional heavyweight metric backends.

    Implemented here (dependency-free): PSNR, SSIM, LPIPS-proxy, simplified
    PPL, KID (polynomial MMD), Inception-Score (requires torchvision inception
    weights), subspace traversal grids.  Clean-FID remains optional via the
    ``cleanfid`` package.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    # ---------------------------- pair-wise metrics ----------------------------

    @staticmethod
    def calculate_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> float:
        """PSNR in dB.

        ``data_range`` defaults to 2.0 for tensors in [-1, 1] (the training
        pipeline's Tanh convention); pass 1.0 for [0, 1] inputs — a wrong
        range inflates/deflates the dB value by a constant offset.
        """
        mse = F.mse_loss(x.float(), y.float()).item()
        if mse <= 1e-12:
            return float("inf")
        return float(10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item())

    @staticmethod
    def _gaussian_window(kernel_size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        window_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        window_1d = (window_1d / window_1d.sum()).unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        return window_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()

    @staticmethod
    def calculate_ssim(
        x: torch.Tensor,
        y: torch.Tensor,
        data_range: float = 2.0,
        kernel_size: int = 11,
    ) -> float:
        """Standard SSIM (Wang et al. 2004) computed on the fly.

        ``data_range`` defaults to 2.0 for tensors in [-1, 1]; pass 1.0 for
        [0, 1] inputs — C1/C2 scale with the range, so a mismatched range
        silently shifts the score (towards 1 when the range is too large).
        """
        x, y = x.float(), y.float()
        channels = x.size(1)
        window = GenAIEvaluator._gaussian_window(kernel_size, channels=channels).to(x.device)

        mu_x = F.conv2d(x, window, padding=kernel_size // 2, groups=channels)
        mu_y = F.conv2d(y, window, padding=kernel_size // 2, groups=channels)
        mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=kernel_size // 2, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=kernel_size // 2, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=kernel_size // 2, groups=channels) - mu_xy

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        return float(ssim_map.mean().item())

    @staticmethod
    def calculate_lpips_proxy(x: torch.Tensor, y: torch.Tensor) -> float:
        """Dependency-free perceptual proxy; use the official ``lpips`` package for papers."""
        pooled_x = F.avg_pool2d(x.float(), kernel_size=4)
        pooled_y = F.avg_pool2d(y.float(), kernel_size=4)
        return float(F.l1_loss(pooled_x, pooled_y).item())

    # --------------------------- distribution metrics ---------------------------

    @staticmethod
    def calculate_kid(
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
        subsets: int = 100,
        max_subset_size: int = 1000,
        *,
        estimator: str = "unbiased",
        clip_negative: bool = True,
    ) -> tuple[float, float]:
        """KID (MMD^2 with the cubic polynomial kernel) per Binkowski et al. 2018.

        Two estimators are offered because they answer different questions:

        * ``"unbiased"`` (default, matches the reference implementation and
          torch-fidelity): the two self terms drop their diagonals and are scaled
          by ``1/(m(m-1))``; the cross term keeps its full sum, scaled by
          ``1/m^2``. This is the published KID estimator. It is **not** zero for
          identical inputs: when both subsets are drawn independently from the
          same finite pool of size ``n``, ``E[estimate] = -2*(mu_self - mu_off)/n``
          because a cross pair lands on the same pool element with probability
          ``1/n`` (the often-quoted ``-O(1/m)`` rate applies only when the *same*
          subset is scored on both sides). torch-fidelity reports this raw value
          and notes that slightly negative KID means very similar distributions.
        * ``"biased"`` (the V-statistic): no diagonal is dropped; all three terms
          use ``1/m^2``. It is a squared RKHS-norm estimate, hence non-negative,
          and is exactly 0 only when both inputs are element-wise the same
          subset; for independently drawn subsets of an identical pool it sits
          slightly *above* zero (bias ~O(1/m)) — see the ordering-contract test.

        Subsets are drawn without replacement from a single fixed-seed generator
        (seed 0), mirroring torch-fidelity's single-RNG-stream pattern, so the
        metric is deterministic. The subset size is clamped to
        ``min(len(real), len(fake), max_subset_size)`` (StyleGAN3-lineage
        behaviour) and the log line reports the effective size.

        ``clip_negative`` (default True) clamps the *reported mean* at 0 — a
        presentation choice; torch-fidelity itself reports the raw value. Set it
        False for raw, reference-comparable output, which may be slightly
        negative for very similar distributions.
        """
        real_features = real_features.float().cpu()
        fake_features = fake_features.float().cpu()
        if real_features.ndim != 2 or fake_features.ndim != 2:
            raise ValueError("features must be 2-D [N, D]")
        if real_features.size(1) != fake_features.size(1):
            raise ValueError("real/fake feature dimension mismatch")
        if not (torch.isfinite(real_features).all() and torch.isfinite(fake_features).all()):
            raise ValueError("features contain non-finite values")
        if subsets < 1:
            raise ValueError("subsets must be >= 1")
        if estimator not in {"unbiased", "biased"}:
            raise ValueError(f"estimator must be 'unbiased' or 'biased', got {estimator!r}")
        n = real_features.size(0)
        m = min(fake_features.size(0), n, max_subset_size)
        if m < 2:
            raise ValueError("KID estimator needs at least 2 samples per set")
        d = real_features.size(1)

        def poly_mmd(a: torch.Tensor, b: torch.Tensor) -> float:
            k_aa = (a @ a.t() / d + 1) ** 3
            k_bb = (b @ b.t() / d + 1) ** 3
            k_ab = (a @ b.t() / d + 1) ** 3
            na, nb = a.size(0), b.size(0)
            if estimator == "biased":
                # V-statistic: every term normalised by n^2, no diagonal removed.
                return float(
                    k_aa.sum() / (na * na)
                    + k_bb.sum() / (nb * nb)
                    - 2 * k_ab.sum() / (na * nb)
                )
            # Unbiased: self terms de-diagonalised with 1/(n(n-1)); the cross term
            # deliberately keeps its diagonal and is normalised by 1/(na*nb).
            return float(
                (k_aa.sum() - torch.diag(k_aa).sum()) / (na * (na - 1))
                + (k_bb.sum() - torch.diag(k_bb).sum()) / (nb * (nb - 1))
                - 2 * k_ab.sum() / (na * nb)
            )

        scores = []
        generator = torch.Generator().manual_seed(0)
        for _ in range(subsets):
            idx_real = torch.randperm(n, generator=generator)[:m]
            idx_fake = torch.randperm(fake_features.size(0), generator=generator)[:m]
            scores.append(poly_mmd(real_features[idx_real], fake_features[idx_fake]))
        mean = sum(scores) / len(scores)
        # Population std (÷N), matching the reference implementations' np.std.
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        raw_mean = mean
        if clip_negative:
            mean = max(mean, 0.0)
        if clip_negative and raw_mean < 0:
            # A clipped report must stay honest: show the raw estimate too,
            # so the log remains comparable with torch-fidelity's output.
            logger.info(
                "KID ({}): mean={:.6f} (raw {:.6f}, clipped at 0) std={:.6f} ({} subsets of {})",
                estimator, mean, raw_mean, std, subsets, m,
            )
        else:
            logger.info(
                "KID ({}): mean={:.6f} std={:.6f} ({} subsets of {})",
                estimator, mean, std, subsets, m,
            )
        return mean, std

    @staticmethod
    def _inception_preprocess(images: torch.Tensor) -> torch.Tensor:
        """Apply the documented Inception_V3_Weights preprocessing to [0,1] images.

        torchvision's ``transform_input`` flag is NOT a substitute for the weight
        transform (resize 342 -> center-crop 299 -> ImageNet mean/std). Use the
        official weight transform and keep the model's ``transform_input=False``.
        """
        from torchvision.models import Inception_V3_Weights

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be N x 3 x H x W, got {tuple(images.shape)}")
        if not torch.isfinite(images).all():
            raise ValueError("images contain non-finite values")
        return Inception_V3_Weights.DEFAULT.transforms()(images.float())

    @staticmethod
    def _load_inception(device: torch.device | str) -> torch.nn.Module:
        """Inception-V3 with the official ImageNet weights, loaded under ``weights_only=True``.

        ``inception_v3(weights=...)`` fetches the state dict through
        ``WeightsEnum.get_state_dict`` -> ``torch.hub.load_state_dict_from_url``,
        whose ``weights_only`` defaults to **False** (torch 2.5.1 and 2.14.1 alike;
        the ``torch.load`` default flip in 2.6 does not reach it because hub passes
        the value explicitly). That would unpickle the cached ``.pth`` unrestricted,
        an escape from the repo-wide ``weights_only=True`` rule that a literal-only
        ``torch.load`` scan cannot see. Build the architecture here exactly as
        torchvision does for ``weights=`` (aux_logits=True, init_weights=False,
        ``transform_input=False`` — the official transform handles preprocessing,
        see :meth:`_inception_preprocess`) and load the state dict ourselves.
        """
        from torchvision.models import Inception_V3_Weights, inception_v3

        from eigenforge.common.torch_guard import require_safe_torch

        require_safe_torch(when="GenAIEvaluator._load_inception")
        model = inception_v3(weights=None, aux_logits=True, transform_input=False, init_weights=False)
        state = Inception_V3_Weights.DEFAULT.get_state_dict(
            progress=True, check_hash=True, weights_only=True
        )
        model.load_state_dict(state)
        return model.to(device).eval()

    @staticmethod
    def calculate_inception_score(images: torch.Tensor, splits: int = 10) -> tuple[float, float]:
        """IS per Salimans et al. 2016 using torchvision InceptionV3 logits.

        ``images`` must be in [0, 1] with shape [N, 3, H, W]; ImageNet
        normalization is applied via :meth:`_inception_preprocess`.
        """
        n = images.size(0)
        if n < 1:
            raise ValueError("need at least one image")
        if splits < 1 or splits > n:
            raise ValueError(f"splits must satisfy 1 <= splits <= N ({n}), got {splits}")

        device = images.device
        model = GenAIEvaluator._load_inception(device)
        with torch.no_grad():
            logits = []
            for chunk in images.split(32):
                logits.append(model(GenAIEvaluator._inception_preprocess(chunk)))
            logits = torch.cat(logits)
        # torch-fidelity protocol alignment: a fixed-seed shuffle before
        # splitting (IS is sensitive to sample order), and a log-softmax KL —
        # float32 softmax can underflow the 1000-way ImageNet distribution to
        # exact zero, where clamping introduced truncation bias; logsumexp
        # stays stable.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(2020))
        logits = logits[order]
        log_probs = logits.log_softmax(dim=1)
        probs = log_probs.exp()
        split_scores = []
        for i in range(splits):
            lo, hi = i * n // splits, (i + 1) * n // splits
            part_probs, part_log = probs[lo:hi], log_probs[lo:hi]
            log_mean = torch.logsumexp(part_log, dim=0, keepdim=True) - math.log(part_probs.size(0))
            kl = (part_probs * (part_log - log_mean)).sum(dim=1)
            split_scores.append(kl.mean().exp().item())
        mean = sum(split_scores) / splits
        # Population std (÷N), matching the reference implementations' np.std
        # and the KID convention in this file.
        std = (sum((s - mean) ** 2 for s in split_scores) / splits) ** 0.5
        logger.info("Inception Score: {:.4f} ± {:.4f}", mean, std)
        return mean, std

    @staticmethod
    def extract_inception_features(images: torch.Tensor) -> torch.Tensor:
        """2048-d pool3 features for FID/KID (documented ImageNet preprocessing)."""
        model = GenAIEvaluator._load_inception(images.device)
        model.fc = torch.nn.Identity()
        feats = []
        with torch.no_grad():
            for chunk in images.split(32):
                feats.append(model(GenAIEvaluator._inception_preprocess(chunk)))
        return torch.cat(feats)

    def compute_clean_fid(self, real_dir: str, fake_dir: str) -> float:
        try:
            from cleanfid import fid
        except ImportError as exc:
            raise RuntimeError("Install the optional 'clean-fid' dependency to compute Clean-FID") from exc
        score = fid.compute_fid(real_dir, fake_dir, device=self.device)
        logger.info("Clean-FID: {:.4f}", score)
        return float(score)

    # ------------------------------ latent metrics ------------------------------

    def compute_ppl(
        self,
        generator: torch.nn.Module,
        num_samples: int = 50,
        epsilon: float = 1e-4,
        per_style_layer: bool = False,
        batch_size: int = 32,
        seed: int = 0,
    ) -> dict[int, float] | float:
        """PROXY Perceptual Path Length: pixel-space MSE per unit latent step.

        Not comparable with StyleGAN's published PPL. Differences from the
        definition (Karras et al. 2019): the distance is plain pixel MSE, not
        LPIPS/VGG16 feature distance; interpolation is *linear* in z (no slerp);
        the path is perturbed by ``epsilon`` along the unit direction of
        ``z2 - z1`` rather than a step of ``t + epsilon`` on the interpolation
        parameter; and no 1%-99% outlier trimming is applied. Use it only to
        compare checkpoints of this same code base under this same protocol.

        - When ``per_style_layer`` is True: isolates the sensitivity of style layer
          ``layer_idx`` by holding the base latent vector ``z`` fixed and perturbing
          only the target layer's subspace coordinates by ``epsilon`` along random
          unit directions. This measures direct layer control rather than non-target drift.
        - When ``per_style_layer`` is False: measures global latent space smoothness
          along linear interpolations.

        Monte-Carlo sampling is drawn from a local ``seed``-ed generator (not the
        global RNG) so results are reproducible, matching the determinism
        discipline of KID/traversal in this module.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        generator.eval()
        generator = generator.to(self.device)
        rng = torch.Generator().manual_seed(seed)

        def _randn(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=rng).to(self.device)

        def _randu(*shape: int) -> torch.Tensor:
            return torch.rand(*shape, generator=rng).to(self.device)

        if per_style_layer:
            per_layer: dict[int, float] = {}
            for layer_idx in range(generator.n_stages):
                total_loss = 0.0
                seen = 0
                for start in range(0, num_samples, batch_size):
                    cur_n = min(batch_size, num_samples - start)
                    z_base = _randn(cur_n, generator.latent_dim)
                    c_base = _randn(cur_n, generator.n_basis)
                    v = _randn(c_base.shape[0], c_base.shape[1])
                    v_unit = F.normalize(v, dim=1)
                    c_perturbed = c_base + epsilon * v_unit

                    with torch.no_grad():
                        img_base = generator(z_base, layer_edits={layer_idx: c_base})
                        img_pert = generator(z_base, layer_edits={layer_idx: c_perturbed})
                    diff = ((img_base - img_pert) / epsilon) ** 2
                    total_loss += float(diff.mean(dim=(1, 2, 3)).sum().item())
                    seen += cur_n
                per_layer[layer_idx] = float(total_loss / max(seen, 1))
            return per_layer

        # Global PPL
        total_loss = 0.0
        seen = 0
        for start in range(0, num_samples, batch_size):
            cur_n = min(batch_size, num_samples - start)
            z1 = _randn(cur_n, generator.latent_dim)
            z2 = _randn(cur_n, generator.latent_dim)
            t = _randu(cur_n, 1)
            z_mid = z1 + t * (z2 - z1)
            z_next = z_mid + epsilon * F.normalize(z2 - z1, dim=1)
            with torch.no_grad():
                img_a = generator(z_mid)
                img_b = generator(z_next)
            diff = ((img_a - img_b) / epsilon) ** 2
            total_loss += float(diff.mean(dim=(1, 2, 3)).sum().item())
            seen += cur_n
        return float(total_loss / max(seen, 1))

    # ------------------------------- visualisation -------------------------------

    def generate_subspace_traversal_grid(
        self,
        generator: torch.nn.Module,
        layer_idx: int,
        basis_idx: int,
        alpha_range: list[float] | None = None,
        seed: int = 42,
        output_path: str | None = None,
    ) -> torch.Tensor:
        alpha_range = alpha_range or [-3.0, -1.5, 0.0, 1.5, 3.0]
        generator = generator.to(self.device).eval()
        generator_n_basis = getattr(generator, "n_basis", None)
        if generator_n_basis is None or not 0 <= basis_idx < generator_n_basis:
            raise IndexError("basis_idx out of range")
        n_stages = getattr(generator, "n_stages", None)
        if n_stages is None or not 0 <= layer_idx < n_stages:
            raise IndexError("layer_idx out of range")
        torch.manual_seed(seed)
        z = torch.randn(1, generator.latent_dim, device=self.device)
        images = []
        with torch.no_grad():
            for value in alpha_range:
                coords = torch.zeros(1, generator_n_basis, device=self.device)
                coords[:, basis_idx] = value
                images.append(generator(z, layer_edits={layer_idx: coords}, strict_qr=True))
        grid = make_grid(torch.cat(images, dim=0), nrow=len(alpha_range), normalize=True, value_range=(-1, 1))
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            save_image(grid, output_path)
        return grid
