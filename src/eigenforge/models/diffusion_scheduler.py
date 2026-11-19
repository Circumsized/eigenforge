import math

import torch
from torch import nn


class DiffusionScheduler(nn.Module):
    """DDPM forward process plus DDPM/DDIM sampling utilities."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
        *,
        timesteps: int | None = None,
        device: str | None = None,
    ):
        super().__init__()
        if timesteps is not None:
            num_timesteps = timesteps
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")
        self.num_timesteps = num_timesteps
        self.timesteps = num_timesteps

        if schedule_type == "linear":
            # Ho et al.'s 1e-4..0.02 endpoints are defined for T=1000. For any other T the
            # reference codebases (openai/improved-diffusion, openai/guided-diffusion,
            # facebookresearch/DiT) rescale both endpoints by 1000/T so that
            # alpha_bar_T stays ~0 and x_T ~ N(0, I); unscaled, T=100 ends at
            # alpha_bar_T ~ 0.36 and sampling from N(0, I) starts off-distribution.
            # Identical to the unscaled schedule at T=1000. The clamp only bites for
            # tiny T (scale*0.02 > 1), where a beta >= 1 would make alpha negative.
            scale = 1000.0 / num_timesteps
            betas = torch.linspace(
                beta_start * scale, beta_end * scale, num_timesteps, dtype=torch.float32
            ).clamp(max=0.9999)
        elif schedule_type == "cosine":
            x = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float32)
            cumulative = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            cumulative = cumulative / cumulative[0]
            betas = 1.0 - cumulative[1:] / cumulative[:-1]
            betas = betas.clamp(1e-4, 0.9999)
        else:
            raise ValueError(f"unknown schedule_type: {schedule_type}")

        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        cumulative_prev = torch.cat([torch.ones(1), cumulative[:-1]])
        posterior_variance = betas * (1.0 - cumulative_prev) / (1.0 - cumulative)

        for name, value in {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": cumulative,
            "alphas_cumprod_prev": cumulative_prev,
            "sqrt_alphas_cumprod": torch.sqrt(cumulative),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - cumulative),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": torch.log(posterior_variance.clamp(min=1e-20)),
        }.items():
            self.register_buffer(name, value)
        if device is not None:
            self.to(device)

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        values = values.to(t.device)
        out = values.gather(0, t.long().clamp(0, values.shape[0] - 1))
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def _normalize_timesteps(self, t: torch.Tensor, size: int, x: torch.Tensor) -> torch.Tensor:
        """Return a validated batch-length LongTensor of timesteps.

        Accepts a 0-D scalar (broadcast to the batch) or a [B] tensor; rejects
        non-integral floats, batch mismatches, and out-of-range indices. The
        returned tensor MUST be used downstream — a prior version reshaped a
        local copy and let _extract see the original 0-D tensor.
        """
        if torch.is_floating_point(t) and not torch.equal(t, t.round()):
            raise ValueError("timesteps must be integer-valued")
        t = t.long()
        if t.ndim == 0:
            t = t.reshape(1).expand(x.shape[0])
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must be scalar or shape [{x.shape[0]}], got shape {tuple(t.shape)}")
        if not bool(torch.all((t >= 0) & (t < size))):
            raised = t[torch.where((t < 0) | (t >= size))]
            raise ValueError(f"timesteps out of range [0, {size - 1}]: {raised.tolist()}")
        return t

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        return_noise: bool = False,
    ):
        """Sample ``x_t`` from q(x_t|x_0); optionally return the sampled noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
        t = self._normalize_timesteps(t.to(x_start.device), self.num_timesteps, x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_t = sqrt_alpha * x_start + sqrt_one_minus * noise
        return (x_t, noise) if return_noise else x_t

    def q_sample_with_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        return self.q_sample(x_start, t, noise, return_noise=True)

    def p_sample(self, pred_noise: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """One stochastic DDPM reverse step given an epsilon prediction."""
        if isinstance(t, int):
            t = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        else:
            t = t.to(x_t.device)
        t = self._normalize_timesteps(t, self.num_timesteps, x_t)
        beta = self._extract(self.betas, t, x_t.shape)
        alpha = self._extract(self.alphas, t, x_t.shape)
        alpha_bar = self._extract(self.alphas_cumprod, t, x_t.shape)
        mean = (x_t - beta * pred_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float().reshape(t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(variance.clamp(min=1e-20)) * noise

    def _build_time_grid(self, steps: int) -> torch.Tensor:
        """Descending strided grid from T-1 down to 0.

        ``unique()`` sorts ascending, so the result is flipped back; the grid
        must strictly decrease so sampling walks the reverse process from the
        noisiest timestep down to 0.
        """
        steps = max(2, min(int(steps), self.num_timesteps))
        grid = torch.linspace(self.num_timesteps - 1, 0, steps).round().long().unique().flip(0)
        if grid[-1].item() != 0:
            grid = torch.cat([grid, torch.zeros(1, dtype=torch.long)])
        # Append -1 as terminal boundary representing clean state (t < 0 => alpha_prev = 1.0)
        grid = torch.cat([grid, torch.tensor([-1], dtype=torch.long)])
        return grid

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        steps: int = 50,
        eta: float = 0.0,
        device: str | None = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """DDIM sampling (Song et al. 2021).

        ``clip_denoised`` clamps the predicted x0 to the data range [-1, 1] and
        re-derives epsilon from the clipped x0, as the reference samplers do
        (``clip_denoised=True`` in improved-diffusion/guided-diffusion,
        ``clip_sample=True`` in diffusers' DDIMScheduler). Without it the first
        step divides the epsilon error by sqrt(alpha_bar_T) (1.6e-5 for the
        cosine schedule), amplifying it ~6e4x.
        """
        if eta < 0:
            raise ValueError("eta must be non-negative")
        if eta > 1:
            raise ValueError("eta must be <= 1 (DDIM interpolates DDIM<->DDPM over [0,1])")
        if steps < 2:
            raise ValueError(
                "steps must be at least 2: the time grid is clamped to a span of "
                "at least 2, so steps=1 would silently execute a different grid "
                "than requested"
            )
        if len(shape) != 4 or any(s <= 0 for s in shape):
            raise ValueError(f"shape must be a positive 4-tuple (B, C, H, W), got {shape}")
        if device is None:
            device = next(model.parameters(), self.betas).device
        time_grid = self._build_time_grid(steps).to(device)
        image = torch.randn(shape, device=device)

        for index, current in enumerate(time_grid[:-1]):
            previous = time_grid[index + 1]
            t = torch.full((shape[0],), int(current.item()), device=device, dtype=torch.long)
            pred_noise = model(image, t)
            alpha_bar = self._extract(self.alphas_cumprod, t, image.shape)
            pred_x0 = (image - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            if clip_denoised:
                pred_x0 = pred_x0.clamp(-1.0, 1.0)
                pred_noise = (image - torch.sqrt(alpha_bar) * pred_x0) / torch.sqrt(1.0 - alpha_bar)

            if int(previous.item()) < 0:
                # Reached timestep 0: alpha_prev = 1.0, variance collapses deterministically to 0
                alpha_prev = torch.ones_like(alpha_bar)
                sigma = torch.zeros_like(alpha_bar)
            else:
                t_prev = torch.full((shape[0],), int(previous.item()), device=device, dtype=torch.long)
                alpha_prev = self._extract(self.alphas_cumprod, t_prev, image.shape)
                # DDIM (Song et al. 2021 Eq.16):
                # sigma = eta * sqrt((1-a_prev)/(1-a_t)) * sqrt(1 - a_t/a_prev).
                # In reverse sampling a_prev > a_t, so 1 - a_t/a_prev > 0.
                sigma = eta * torch.sqrt(((1.0 - alpha_prev) / (1.0 - alpha_bar)).clamp(min=0)) * torch.sqrt(
                    (1.0 - alpha_bar / alpha_prev).clamp(min=0)
                )

            direction = torch.sqrt((1.0 - alpha_prev - sigma.square()).clamp(min=0)) * pred_noise
            image = torch.sqrt(alpha_prev) * pred_x0 + direction
            if eta > 0 and int(previous.item()) >= 0:
                image = image + sigma * torch.randn_like(image)
        return image

    @torch.no_grad()
    def sample(self, model: nn.Module, shape, steps: int = 50, eta: float = 0.0, device=None, clip_denoised: bool = True):
        return self.ddim_sample(model, shape, steps=steps, eta=eta, device=device, clip_denoised=clip_denoised)
