"""GPU/CPU 冒烟 CLI（Slice 6）。

在目标设备上执行 EigenGAN 与扩散管线的真实步进冒烟，落盘结构化报告。
无 CUDA 时报告如实标注 ``cuda_available=false`` 与 ``quality_gate=
"not_evaluated"``——质量门禁只由固定硬件 GPU runner 评估，本工具绝不
伪造 GPU 指标。用法::

    python eigenforge/scripts/gpu_smoke.py [--seeds 0,1,2] [--steps 2] [--output outputs/gpu_smoke.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eigenforge.common.fsutil import atomic_write_json
from eigenforge.common.logger import logger
from eigenforge.smoke import run_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="training-pipeline smoke with honest device report")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds (default: 0,1,2)")
    parser.add_argument("--steps", type=int, default=2, help="optimization steps per model per seed")
    parser.add_argument(
        "--output", default="outputs/gpu_smoke.json", help="report destination (default: outputs/gpu_smoke.json)"
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit 2 instead of running when no CUDA device is available (use on GPU runners so a "
        "silent CPU fallback cannot be mistaken for a GPU verification)",
    )
    args = parser.parse_args(argv)

    try:
        seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())
    except ValueError:
        print("FAIL: --seeds must be comma-separated integers", file=sys.stderr)
        return 2
    if args.steps < 1:
        print("FAIL: --steps must be positive", file=sys.stderr)
        return 2
    if args.require_cuda and not torch.cuda.is_available():
        print("FAIL: --require-cuda was given but no CUDA device is available", file=sys.stderr)
        return 2

    try:
        report = run_smoke(seeds=seeds, steps=args.steps)
    except Exception as exc:  # 冒烟失败必须如实失败退出
        logger.error("smoke protocol failed: {}: {}", type(exc).__name__, exc)
        print(f"FAIL: smoke protocol failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    atomic_write_json(args.output, report)

    device = report["device"]
    print(
        f"SMOKE OK: device={device} cuda_available={report['cuda_available']} "
        f"seeds={report['seeds']} steps={report['steps_per_seed']} "
        f"quality_gate={report['quality_gate']} report={args.output}"
    )
    if not report["cuda_available"]:
        print(
            "  NOTE: pipeline-only check on CPU — this is NOT a GPU/AMP verification and "
            "says nothing about model quality."
        )
    for entry in report["aggregate"].items():
        name, stats = entry
        print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
