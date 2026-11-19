"""Minimal reproducible EigenGAN training entry point.

Supports deterministic seeding, AMP on CUDA, and checkpoint resume.
"""
import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eigenforge.checkpoints import (
    ResumeState,
    check_finite_losses,
    check_finite_parameters,
    load_checkpoint,
    save_checkpoint,
)
from eigenforge.common.logger import logger
from eigenforge.common.torch_guard import enforce_safe_torch_at_startup
from eigenforge.models.eigengan import (
    EigenDiscriminator,
    EigenGenerator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    r1_penalty,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data/celeba")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--n-basis", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ortho-weight", type=float, default=1.0)
    parser.add_argument("--r1-gamma", type=float, default=10.0)
    parser.add_argument(
        "--d-reg-every",
        type=int,
        default=16,
        help="Lazy R1 regularization frequency in steps (StyleGAN2/EigenGAN best practice to save ~40%% D runtime)",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--output-dir", default="./experiments/eigengan")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to latest.pt to resume training")
    parser.add_argument(
        "--allow-unverified-resume",
        action="store_true",
        help="Permit resuming a checkpoint that has no sibling manifest (no sha256/architecture "
        "verification). Off by default: an unverifiable resume can silently mix schedules/shapes.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision autocast")
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


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)


def ema_decay_at(step: int, decay: float) -> float:
    """EMA decay with the standard warm-up ``min(decay, (1 + n) / (10 + n))``.

    Same ramp (and same reason) as ``train_diffusion.ema_decay_at``: TensorFlow's
    ``tf.train.ExponentialMovingAverage(num_updates=...)`` uses exactly this
    formula. A constant 0.999 starting from random initialization leaves
    ``0.999**n`` of the *initial* weights in the average (about 37% after
    1,000 steps, and ~0.18% after a 6.3k-step CelebA epoch at batch 32), and
    the EMA weights are what sampling and eval consume — long averages must
    not carry the random init (Karras et al., CVPR 2024, Sec. 3). EigenGAN's EMA stays generator-only (the discriminator
    has none, per the model card); this only changes the decay schedule.
    """
    return min(decay, (1.0 + step) / (10.0 + step))


def discriminator_loss_with_lazy_r1(
    discriminator: torch.nn.Module,
    generator: torch.nn.Module,
    real: torch.Tensor,
    z: torch.Tensor,
    *,
    r1_gamma: float,
    d_reg_every: int,
    step: int,
    autocast_device: str | None = None,
):
    """Hinge D loss; the R1 second-order graph exists ONLY on scheduled steps.

    Kept as a unit so the lazy-R1 contract (compute ``do_r1`` first, build
    ``real.requires_grad_`` + the extra D forward inside the branch) is
    testable without a dataset. Lazy R1 matches the original notebook and
    StyleGAN2 practice and saves roughly 40% of discriminator compute.
    ``autocast_device`` enables AMP autocast; ``None`` runs in full precision
    (an ``enabled=False`` autocast context is a documented no-op, so skipping
    it outright is equivalent).
    """
    with torch.no_grad():
        fake = generator(z)
    if autocast_device:
        with torch.autocast(device_type=autocast_device):
            d_loss = discriminator_hinge_loss(discriminator(real), discriminator(fake))
    else:
        d_loss = discriminator_hinge_loss(discriminator(real), discriminator(fake))

    do_r1 = r1_gamma > 0 and (d_reg_every <= 1 or step % d_reg_every == 0)
    if do_r1:
        real_for_r1 = real.detach().requires_grad_(True)
        real_logits_for_r1 = discriminator(real_for_r1)
        gamma = r1_gamma * (d_reg_every if d_reg_every > 1 else 1)
        d_loss = d_loss + gamma * r1_penalty(real_logits_for_r1, real_for_r1)
    return d_loss, fake


def apply_resume_state(
    *,
    generator: torch.nn.Module,
    ema_generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    g_optim: torch.optim.Optimizer,
    d_optim: torch.optim.Optimizer,
    resume_state: ResumeState,
    path: str,
) -> int:
    """Wire a verified checkpoint into the trainables; return the start epoch.

    Checkpoints from before EMA tracking legitimately fall back to the raw
    generator weights, but silently resuming would make all EMA-based
    sampling/eval run on a pseudo-EMA nobody asked for — hence the loud
    warning (commit d9f7df4's contract) instead of a silent fallback.
    """
    ckpt = resume_state.payload
    generator.load_state_dict(ckpt["generator"])
    if "ema_generator" in ckpt:
        ema_generator.load_state_dict(ckpt["ema_generator"])
    else:
        logger.warning(
            "Checkpoint {} lacks ema_generator; seeding EMA from raw generator weights.",
            path,
        )
        ema_generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    g_optim.load_state_dict(ckpt["g_optim"])
    d_optim.load_state_dict(ckpt["d_optim"])
    return resume_state.epoch


def main():
    args = parse_args()
    # After parse_args so --help works on a vulnerable torch; before anything
    # loads checkpoints.
    enforce_safe_torch_at_startup(when="train_eigengan")
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
    generator = EigenGenerator(args.image_size, args.z_dim, args.n_basis).to(device)
    ema_generator = copy.deepcopy(generator).eval()
    for parameter in ema_generator.parameters():
        parameter.requires_grad_(False)
    discriminator = EigenDiscriminator(args.image_size).to(device)
    g_optim = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_epoch = 0
    if args.resume:
        # Manifest-verified resume: sha256 against the sibling manifest, epoch
        # schema enforced, and the recorded architecture must match this run.
        resume_state = load_checkpoint(
            args.resume,
            map_location=str(device),
            required_keys=("generator", "discriminator", "g_optim", "d_optim"),
            expected_config={
                "z_dim": args.z_dim,
                "image_size": args.image_size,
                "n_basis": args.n_basis,
            },
            require_manifest=not args.allow_unverified_resume,
            fallback_to_previous=True,
        )
        start_epoch = apply_resume_state(
            generator=generator,
            ema_generator=ema_generator,
            discriminator=discriminator,
            g_optim=g_optim,
            d_optim=d_optim,
            resume_state=resume_state,
            path=args.resume,
        )
        logger.info(
            "Resumed from {} at epoch {} (manifest verified={})",
            args.resume,
            start_epoch,
            resume_state.manifest is not None,
        )

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for epoch in range(start_epoch, args.epochs):
        for step, (real, _) in enumerate(loader):
            real = real.to(device)
            z = torch.randn(real.size(0), args.z_dim, device=device)
            d_loss, fake = discriminator_loss_with_lazy_r1(
                discriminator,
                generator,
                real,
                z,
                r1_gamma=args.r1_gamma,
                d_reg_every=args.d_reg_every,
                step=step,
                autocast_device=device.type if use_amp else None,
            )
            check_finite_losses({"d_loss": d_loss}, epoch=epoch, step=step)
            d_optim.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(d_optim)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                fake_logits = discriminator(generator(z))
                g_loss = generator_hinge_loss(fake_logits)
                total_loss = g_loss + args.ortho_weight * generator.total_orthogonality_loss()

            check_finite_losses(
                {"g_loss": g_loss, "total_loss": total_loss}, epoch=epoch, step=step
            )
            g_optim.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(g_optim)
            scaler.update()
            # Warm-up ramp on the global step (epoch * steps + step), identical
            # in form to train_diffusion: the EMA must not carry the random
            # initialization into long averages.
            update_ema(
                ema_generator, generator, ema_decay_at(epoch * len(loader) + step, args.ema_decay)
            )

            if step % 100 == 0:
                logger.info("epoch={} step={} d_loss={:.4f} g_loss={:.4f}", epoch, step, d_loss.item(), g_loss.item())
                save_image((fake[:16].float() + 1) / 2, output_dir / f"sample_{epoch:03d}_{step:06d}.png", nrow=4)

        # The per-step breaker only sees pre-update losses; the epoch's final
        # optimizer step must not reach disk with NaN/Inf weights.
        check_finite_parameters(
            {"generator": generator, "ema_generator": ema_generator, "discriminator": discriminator},
            epoch=epoch,
        )
        # Durable atomic write + sibling manifest; the previous verified
        # generation is kept as latest.pt.prev for crash recovery.
        save_checkpoint(
            output_dir / "latest.pt",
            {
                "generator": generator.state_dict(),
                "ema_generator": ema_generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optim": g_optim.state_dict(),
                "d_optim": d_optim.state_dict(),
                "epoch": epoch + 1,
            },
            model_kind="eigengan",
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


if __name__ == "__main__":
    main()
