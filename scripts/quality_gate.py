"""三随机种子质量门禁协议（Slice 8）。

``--mode protocol``（默认）：固定合成协议下对每个 seed 执行 tiny 短训练并
采集不依赖网络下载的指标（GAN 生成器损失、EigenGAN 正交误差、扩散损失），
跨种子聚合 mean/std——只证明**协议可执行**，不做任何质量判定。

``--mode evaluate``：对同一协议指标施加显式阈值判定；未提供阈值即拒判
（退出 2），绝不默认通过。判定是失败封闭的：非有限（NaN/inf）的阈值被拒绝
（退出 2），非有限的指标计为越界；越界在任何设备上都判 ``fail``（退出 1）；
未越界但无 CUDA 时只报 ``not_evaluated``（退出 2）——CPU 上的 tiny 合成协议
不能认证质量，只有 CUDA 上未越界才判 ``pass``。真实数据集上的质量门禁
（FID 等）仍属固定硬件 GPU runner 的职责——本工具在任何模式下都不冒充
论文可比指标。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eigenforge.artifacts import git_metadata
from eigenforge.common.fsutil import atomic_write_json
from eigenforge.common.logger import logger
from eigenforge.models.diffusion_scheduler import DiffusionScheduler
from eigenforge.models.eigengan import (
    EigenDiscriminator,
    EigenGenerator,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from eigenforge.models.unet_diffusion import UNetDiffusion
from eigenforge.smoke import SmokeCheckError

REPORT_SCHEMA_VERSION = 1
METRIC_KEYS = ("eigengan_loss_g", "eigengan_ortho_error", "diffusion_loss")
THRESHOLD_FLAGS = {
    "eigengan_loss_g": "max_eigengan_loss_g",
    "eigengan_ortho_error": "max_ortho_error",
    "diffusion_loss": "max_diffusion_loss",
}


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_seed(seed: int, train_steps: int, device: torch.device) -> dict[str, Any]:
    _seed_everything(seed)
    generator = EigenGenerator(
        img_size=8, latent_dim=32, n_basis=2, base_channels=8, max_channels=16
    ).to(device)
    discriminator = EigenDiscriminator(img_size=8, base_channels=16).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.99))

    loss_g = None
    for _ in range(train_steps):
        real = torch.rand(2, 3, 8, 8, device=device) * 2.0 - 1.0
        z = torch.randn(2, 32, device=device)
        fake = generator(z)
        opt_d.zero_grad(set_to_none=True)
        loss_d = discriminator_hinge_loss(discriminator(real), discriminator(fake.detach()))
        loss_d.backward()
        opt_d.step()
        opt_g.zero_grad(set_to_none=True)
        loss_g = generator_hinge_loss(discriminator(generator(z)))
        loss_g.backward()
        opt_g.step()
    if loss_g is None or not torch.isfinite(loss_g).all():
        # Explicit raise, not assert: python -O strips asserts, and a NaN loss
        # flowing into evaluate mode compares False against every threshold,
        # reporting quality_gate="pass" for an exploded run.
        raise SmokeCheckError("GAN quality-gate loss is not finite")

    diffusion = UNetDiffusion(base_channels=16).to(device)
    scheduler = DiffusionScheduler(num_timesteps=100).to(device)
    opt_u = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    diffusion_loss = None
    for _ in range(train_steps):
        clean = torch.rand(2, 3, 8, 8, device=device)
        t = torch.randint(0, scheduler.num_timesteps, (2,), device=device)
        noisy, noise = scheduler.q_sample_with_noise(clean, t)
        opt_u.zero_grad(set_to_none=True)
        diffusion_loss = torch.nn.functional.mse_loss(diffusion(noisy, t), noise)
        diffusion_loss.backward()
        opt_u.step()
    if diffusion_loss is None or not torch.isfinite(diffusion_loss).all():
        raise SmokeCheckError("diffusion quality-gate loss is not finite")

    with torch.no_grad():
        ortho_error = float(generator.total_orthogonality_loss().item())

    return {
        "seed": int(seed),
        "eigengan_loss_g": float(loss_g.item()),
        "eigengan_ortho_error": ortho_error,
        "diffusion_loss": float(diffusion_loss.item()),
    }


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [entry[key] for entry in per_seed]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stats[key] = {
            "mean": mean,
            "std": variance ** 0.5,
            "min": min(values),
            "max": max(values),
        }
    return stats


def run_gate(
    seeds: tuple[int, ...],
    train_steps: int,
    mode: str,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    if mode not in {"protocol", "evaluate"}:
        raise ValueError(f"mode must be protocol|evaluate, got {mode!r}")
    if train_steps < 1:
        raise ValueError("train_steps must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed = [_run_seed(seed, train_steps, device) for seed in seeds]
    aggregate = _aggregate(per_seed)
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "mode": mode,
        "dataset": "tiny-synthetic",
        "protocol_executed": True,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "seeds": [int(s) for s in seeds],
        "train_steps": train_steps,
        "git": git_metadata(),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "thresholds": dict(thresholds or {}),
    }

    if mode == "protocol":
        report["quality_gate"] = "not_evaluated"
        return report

    if not thresholds:
        raise ValueError("evaluate mode requires explicit thresholds")
    bad_limits = {flag: v for flag, v in thresholds.items() if not math.isfinite(v)}
    if bad_limits:
        # `mean > nan` is always False, so a NaN/inf limit would wave every run through.
        raise ValueError(f"thresholds must be finite numbers, got {bad_limits}")
    breaches = []
    for metric, flag in THRESHOLD_FLAGS.items():
        limit = thresholds.get(flag)
        if limit is None:
            continue
        mean = aggregate[metric]["mean"]
        # Fail-closed: the metric must itself be finite (-inf satisfies `mean <= limit`
        # and NaN fails `mean > limit`, so a bare comparison passes garbage either way),
        # and is written as "not within limit" so NaN is a breach, not a pass.
        if not math.isfinite(mean):
            breaches.append(f"{metric}: mean {mean} is not finite")
        elif not (mean <= limit):
            breaches.append(f"{metric}: mean {mean:.6f} > threshold {limit:.6f}")
    report["breaches"] = breaches
    if breaches:
        # A breach is informative on any device: the protocol itself diverged.
        report["quality_gate"] = "fail"
    elif not torch.cuda.is_available():
        # Staying inside thresholds on a tiny synthetic CPU protocol certifies
        # nothing about quality; claiming "pass" would contradict smoke.py's
        # "no CUDA => not_evaluated" contract (see AGENTS.md).
        report["quality_gate"] = "not_evaluated"
        report["not_evaluated_reason"] = (
            "no CUDA device: a CPU run of the tiny synthetic protocol cannot certify quality"
        )
    else:
        report["quality_gate"] = "pass"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="three-seed quality gate protocol")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--mode", choices=("protocol", "evaluate"), default="protocol")
    parser.add_argument("--max-eigengan-loss-g", type=float, default=None)
    parser.add_argument("--max-ortho-error", type=float, default=None)
    parser.add_argument("--max-diffusion-loss", type=float, default=None)
    parser.add_argument("--output", default="outputs/quality_gate.json")
    args = parser.parse_args(argv)

    try:
        seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())
    except ValueError:
        print(
            f"FAIL: --seeds must be a comma-separated list of integers, got: {args.seeds!r}",
            file=sys.stderr,
        )
        return 2
    if not seeds:
        print("FAIL: --seeds resolved to an empty seed set", file=sys.stderr)
        return 2
    thresholds = {
        flag: value
        for flag, value in {
            "max_eigengan_loss_g": args.max_eigengan_loss_g,
            "max_ortho_error": args.max_ortho_error,
            "max_diffusion_loss": args.max_diffusion_loss,
        }.items()
        if value is not None
    }
    if args.mode == "evaluate" and not thresholds:
        print(
            "FAIL: evaluate mode requires at least one explicit threshold "
            "(--max-eigengan-loss-g / --max-ortho-error / --max-diffusion-loss); "
            "refusing to judge without one",
            file=sys.stderr,
        )
        return 2
    non_finite = {flag: value for flag, value in thresholds.items() if not math.isfinite(value)}
    if non_finite:
        print(
            f"FAIL: thresholds must be finite numbers (a NaN/inf limit would pass everything), got {non_finite}",
            file=sys.stderr,
        )
        return 2

    started = time.perf_counter()
    try:
        report = run_gate(seeds, args.train_steps, args.mode, thresholds)
    except Exception as exc:
        logger.error("quality gate protocol failed: {}: {}", type(exc).__name__, exc)
        print(f"FAIL: quality gate protocol failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)

    out = Path(args.output)
    atomic_write_json(out, report)

    gate = report["quality_gate"]
    print(
        f"QUALITY GATE: mode={args.mode} gate={gate} seeds={report['seeds']} "
        f"dataset={report['dataset']} report={out}"
    )
    for breach in report.get("breaches", []):
        print(f"  breach: {breach}")
    if gate == "fail":
        return 1
    if args.mode == "evaluate" and gate != "pass":
        # evaluate was explicitly requested but could not certify (e.g. no CUDA):
        # exit non-zero so a CI step cannot read "refused to judge" as success.
        print(f"  not certified: {report.get('not_evaluated_reason', gate)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
