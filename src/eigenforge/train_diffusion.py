"""Minimal reproducible DDPM training entry point.

Supports deterministic seeding, EMA weights for sampling, and resume.
"""
import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eigenforge.checkpoints import (
    check_finite_losses,
    check_finite_parameters,
    load_checkpoint,
    save_checkpoint,
)
from eigenforge.common.logger import logger
from eigenforge.common.torch_guard import enforce_safe_torch_at_startup
from eigenforge.models.diffusion_scheduler import DiffusionScheduler
from eigenforge.models.unet_diffusion import UNetDiffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data/celeba")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--output-dir", default="./experiments/diffusion")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to latest.pt to resume")
    parser.add_argument(
        "--allow-unverified-resume",
        action="store_true",
        help="Permit resuming a checkpoint that has no sibling manifest (no sha256/architecture "
        "verification). Off by default: an unverifiable resume can silently mix schedules/shapes.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic kernels (torch.use_deterministic_algorithms(True), "
        "cudnn.deterministic=True, benchmark=False). Requires "
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 exported before launch; slower. Bitwise "
        "reproducibility still only holds within the same torch version and device "
        "(PyTorch reproducibility notes).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_determinism() -> None:
    """Request deterministic kernels (opt-in via ``--deterministic``).

    ``use_deterministic_algorithms(True)`` raises for ops without a
    deterministic implementation — that is intended: a loud failure beats
    silently drifting results. cudnn.benchmark must be off, otherwise
    autotuning picks different kernels between runs.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(root: str, image_size: int):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    return datasets.ImageFolder(root, transform=transform)


def ema_decay_at(step: int, decay: float) -> float:
    """EMA decay with the standard warm-up ``min(decay, (1 + n) / (10 + n))``.

    TensorFlow's ``tf.train.ExponentialMovingAverage(num_updates=...)`` uses
    exactly this ramp. A constant 0.9999 starting from random initialization
    leaves ``0.9999**n`` of the initial weights in the average (28% after one
    12.7k-step CelebA epoch), and the EMA weights are what the per-epoch
    preview and sampling use; long averages must not carry the random init
    (Karras et al., CVPR 2024, Sec. 3).
    """
    return min(decay, (1.0 + step) / (10.0 + step))


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def main():
    args = parse_args()
    # After parse_args so --help works on a vulnerable torch; before anything
    # loads checkpoints.
    enforce_safe_torch_at_startup(when="train_diffusion")
    seed_everything(args.seed)
    if args.deterministic:
        apply_determinism()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda" and not args.no_amp

    dataset = build_dataset(args.data_root, args.image_size)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    if len(loader) == 0:
        # drop_last with fewer images than one batch yields zero steps; without
        # this the loop would run empty and still save an "epoch+1" checkpoint.
        raise ValueError(
            f"dataset at {args.data_root!r} has {len(dataset)} images, fewer than "
            f"--batch-size {args.batch_size}; nothing to train on"
        )
    model = UNetDiffusion().to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    scheduler = DiffusionScheduler(num_timesteps=args.num_timesteps, schedule_type=args.schedule).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 0
    if args.resume:
        # Manifest-verified resume: sha256 against the sibling manifest, epoch
        # schema enforced, and the recorded architecture must match this run.
        resume_state = load_checkpoint(
            args.resume,
            map_location=str(device),
            required_keys=("model", "ema_model", "optimizer"),
            expected_config={
                "num_timesteps": args.num_timesteps,
                "image_size": args.image_size,
                "schedule": args.schedule,
            },
            require_manifest=not args.allow_unverified_resume,
            fallback_to_previous=True,
        )
        ckpt = resume_state.payload
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = resume_state.epoch
        logger.info(
            "Resumed from {} at epoch {} (manifest verified={})",
            args.resume,
            start_epoch,
            resume_state.manifest is not None,
        )

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for epoch in range(start_epoch, args.epochs):
        for step, (clean, _) in enumerate(loader):
            clean = clean.to(device)
            t = torch.randint(0, scheduler.num_timesteps, (clean.size(0),), device=device)
            noisy, noise = scheduler.q_sample(clean, t, return_noise=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(noisy, t)
                loss = F.mse_loss(pred, noise)
            check_finite_losses({"loss": loss}, epoch=epoch, step=step)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, ema_decay_at(epoch * len(loader) + step, args.ema_decay))
            if step % 100 == 0:
                logger.info("epoch={} step={} loss={:.4f}", epoch, step, loss.item())

        # The per-step breaker only sees pre-update losses; the epoch's final
        # optimizer step must not reach disk with NaN/Inf weights.
        check_finite_parameters({"model": model, "ema_model": ema_model}, epoch=epoch)
        # Durable atomic write + sibling manifest; the previous verified
        # generation is kept as latest.pt.prev for crash recovery. Saved
        # BEFORE the preview sampling so a sampling failure cannot lose the epoch.
        save_checkpoint(
            output_dir / "latest.pt",
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            model_kind="diffusion",
            seed=args.seed,
            step=(epoch + 1) * len(loader),
            training_config={
                **vars(args),
                "torch": torch.__version__,
                "deterministic": args.deterministic,
                "device": str(device),
            },
            dataset={
                "name": "celeba",
                "root": str(args.data_root),
                "num_images": len(dataset),
            },
        )

        samples = scheduler.sample(ema_model, (16, 3, args.image_size, args.image_size), steps=50, device=device)
        save_image((samples.clamp(-1, 1) + 1) / 2, output_dir / f"sample_{epoch:03d}.png", nrow=4)


if __name__ == "__main__":
    main()
