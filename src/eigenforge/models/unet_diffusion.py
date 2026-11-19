import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=t.device) * -scale)
        values = t.float().view(-1, 1) * frequencies.view(1, -1)
        return torch.cat((values.sin(), values.cos()), dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        groups = min(8, in_channels)
        while in_channels % groups:
            groups -= 1
        out_groups = min(8, out_channels)
        while out_channels % out_groups:
            out_groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time(time_embedding)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class UNetDiffusion(nn.Module):
    """Small time-conditioned epsilon-prediction U-Net for 64px faces."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if base_channels % 2 != 0:
            raise ValueError("base_channels must be even (sinusoidal embedding uses dim/2 pairs)")
        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
        )
        self.down1 = ResBlock(in_channels, base_channels, time_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.mid = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.up2 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.up1 = ResBlock(base_channels * 2 + base_channels, base_channels, time_dim)
        out_groups = min(8, base_channels)
        while base_channels % out_groups:
            out_groups -= 1
        self.out = nn.Sequential(nn.GroupNorm(out_groups, base_channels), nn.SiLU(), nn.Conv2d(base_channels, out_channels, 1))
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be 4-D [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[-1] % 4 or x.shape[-2] % 4:
            raise ValueError(f"spatial dims must be divisible by 4, got {tuple(x.shape[-2:])}")
        temb = self.time_embedding(t)
        d1 = self.down1(x, temb)
        d2 = self.down2(self.pool(d1), temb)
        mid = self.mid(self.pool(d2), temb)
        u2 = self.up(mid)
        u2 = self.up2(torch.cat((u2, d2), dim=1), temb)
        u1 = self.up(u2)
        u1 = self.up1(torch.cat((u1, d1), dim=1), temb)
        return self.out(u1)
