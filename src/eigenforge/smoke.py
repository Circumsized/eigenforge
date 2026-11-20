"""GPU/CPU 冒烟协议（Slice 6）。

对 EigenGAN 与 DDPM/DDIM 管线各执行真实的"前向 + 反向 + 更新"步进（tiny
配置），断言 loss 与采样输出无 NaN/Inf，记录耗时（与 CUDA 内存峰值）。
本协议验证的是**训练管线在目标设备上可执行且数值健康**；生成质量门禁
（FID 等）只在固定硬件的 GPU runner 上评估——无 CUDA 时报告如实标注
``quality_gate="not_evaluated"``，绝不伪造 GPU 指标。

tiny 配置刻意与生产容量不同：冒烟只证可执行性，不承诺任何训练质量。
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from eigenforge.artifacts import git_metadata
from eigenforge.models.diffusion_scheduler import DiffusionScheduler
from eigenforge.models.eigengan import (
    EigenDiscriminator,
    EigenGenerator,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from eigenforge.models.unet_diffusion import UNetDiffusion

REPORT_SCHEMA_VERSION = 1
TINY_IMG_SIZE = 8
TINY_BATCH = 2


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


class SmokeCheckError(RuntimeError):
    """A numerical-health check failed during the smoke protocol.

    Raised explicitly rather than via ``assert`` because ``assert`` statements are
    removed under ``python -O``/``PYTHONOPTIMIZE``, which would let an exploded
    (NaN/Inf) run report ``loss_finite: true`` — exactly the fabricated-metric
    failure mode the smoke protocol exists to prevent.
    """


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _smoke_eigengan(device: torch.device, steps: int) -> dict[str, Any]:
    """真实 GAN 步进：D 步（hinge）+ G 步，随后确定性采样。"""
    started = time.perf_counter()
    generator = EigenGenerator(
        img_size=TINY_IMG_SIZE, latent_dim=32, n_basis=2,
        base_channels=8, max_channels=16,
    ).to(device)
    discriminator = EigenDiscriminator(img_size=TINY_IMG_SIZE, base_channels=16).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.99))

    loss_g_value = None
    for _ in range(steps):
        real = torch.rand(TINY_BATCH, 3, TINY_IMG_SIZE, TINY_IMG_SIZE, device=device) * 2.0 - 1.0
        z = torch.randn(TINY_BATCH, 32, device=device)
        fake = generator(z)

        opt_d.zero_grad(set_to_none=True)
        real_logits = discriminator(real)
        fake_logits = discriminator(fake.detach())
        loss_d = discriminator_hinge_loss(real_logits, fake_logits)
        loss_d.backward()
        opt_d.step()

        opt_g.zero_grad(set_to_none=True)
        loss_g = generator_hinge_loss(discriminator(generator(z)))
        loss_g.backward()
        opt_g.step()
        loss_g_value = loss_g.detach()

    if loss_g_value is None or not _finite(loss_g_value):
        raise SmokeCheckError("GAN smoke loss is not finite")
    with torch.no_grad():
        sample = generator(generator.sample_latents(1))
    if not _finite(sample):
        raise SmokeCheckError("GAN smoke sample is not finite")
    return {
        "steps_executed": steps,
        "loss_g": float(loss_g_value.item()),
        "loss_finite": True,
        "sample_shape": list(sample.shape),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _smoke_diffusion(device: torch.device, steps: int) -> dict[str, Any]:
    """真实去噪步进 + 短网格 DDIM 采样。"""
    started = time.perf_counter()
    model = UNetDiffusion(base_channels=16).to(device)
    scheduler = DiffusionScheduler(num_timesteps=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_value = None
    for _ in range(steps):
        # [-1, 1] like the GAN half above and the production trainers
        # (train_diffusion.py normalises with mean/std 0.5), so the sampler's
        # x0 clamp assumption holds and this half exercises the real range.
        clean = torch.rand(TINY_BATCH, 3, TINY_IMG_SIZE, TINY_IMG_SIZE, device=device) * 2.0 - 1.0
        t = torch.randint(0, scheduler.num_timesteps, (TINY_BATCH,), device=device)
        noisy, noise = scheduler.q_sample_with_noise(clean, t)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(noisy, t), noise)
        loss.backward()
        optimizer.step()
        loss_value = loss.detach()

    if loss_value is None or not _finite(loss_value):
        raise SmokeCheckError("diffusion smoke loss is not finite")
    sample = scheduler.sample(model, (1, 3, TINY_IMG_SIZE, TINY_IMG_SIZE), steps=4, device=str(device))
    if not _finite(sample):
        raise SmokeCheckError("diffusion smoke sample is not finite")
    return {
        "steps_executed": steps,
        "loss": float(loss_value.item()),
        "loss_finite": True,
        "sample_shape": list(sample.shape),
        "elapsed_seconds": time.perf_counter() - started,
    }


def aggregate_seed_metrics(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """跨种子聚合标量指标（mean/std/min/max）。"""
    aggregates: dict[str, list[float]] = {"eigengan_loss_g": [], "diffusion_loss": []}
    for entry in per_seed:
        if "eigengan" in entry and "loss_g" in entry["eigengan"]:
            aggregates["eigengan_loss_g"].append(float(entry["eigengan"]["loss_g"]))
        if "diffusion" in entry and "loss" in entry["diffusion"]:
            aggregates["diffusion_loss"].append(float(entry["diffusion"]["loss"]))
    stats: dict[str, dict[str, float]] = {}
    for name, values in aggregates.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stats[name] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
        }
    return stats


def run_smoke(
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    """执行冒烟协议并返回结构化报告。

    无 CUDA 时在 CPU 上完整执行并如实标注；质量门禁恒为
    ``not_evaluated``（由固定硬件 GPU runner 另行评估）。
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not seeds:
        raise ValueError("seeds must not be empty")

    cuda_available = torch.cuda.is_available()
    resolved = torch.device(device) if device else torch.device("cuda" if cuda_available else "cpu")

    per_seed: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for seed in seeds:
        _seed_everything(seed)
        peak_memory: int | None = None
        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)
        started = time.perf_counter()
        eigengan = _smoke_eigengan(resolved, steps)
        diffusion = _smoke_diffusion(resolved, steps)
        elapsed = time.perf_counter() - started
        if resolved.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(resolved))
        per_seed.append({
            "seed": int(seed),
            "eigengan": eigengan,
            "diffusion": diffusion,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_memory,
        })
    elapsed_total = time.perf_counter() - total_started

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "cuda_available": cuda_available,
        "device": str(resolved),
        "quality_gate": "not_evaluated",
        "steps_per_seed": steps,
        "seeds": [int(seed) for seed in seeds],
        "git": git_metadata(),
        "per_seed": per_seed,
        "aggregate": aggregate_seed_metrics(per_seed),
        "elapsed_total_seconds": elapsed_total,
    }
