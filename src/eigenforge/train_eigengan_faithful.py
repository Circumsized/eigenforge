"""EigenGAN training entry point that follows the official TensorFlow trainer.

Mirrors ``train.py`` of LynnHo/EigenGAN-Tensorflow (commit fc3e1a2, MIT): hinge_v1 loss,
zero-centered gradient penalty on real samples (weight 10, computed every 3rd D step and
multiplied by 3), orthogonality weight 1, Adam(lr 1e-4, beta1 0.5), EMA 0.999, one D step
per G step, an "epoch" of ``len(dataset) // batch // 2`` iterations. It trains
``models/eigengan_faithful.py``, not the simplified ``EigenGenerator``.

Never executed. Differences from the official trainer:

* fp32 only. The official run is fp32, and AMP combined with a double-backward penalty is
  exactly the combination this entry point avoids.
* ``--data-root`` is a flat folder of images, like the official ``img_dir``. The official
  data are CelebA images aligned and cropped with its ``scripts/align.py`` (face_factor
  0.45); feeding raw 178x218 CelebA through the square resize below distorts the faces.

Like the official trainer, ``--orth-weight 0`` switches the generator to Gram-Schmidt
orthonormalization of ``U`` in the forward pass, and the learning rate is fixed until
``--epoch-start-decay`` and then decays linearly (``tl.LinearDecayLR``); with the default
``--epoch-start-decay 160 == --epochs`` it never decays.
"""
import argparse
import copy
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
from eigenforge.models.eigengan_faithful import (
    ARCHITECTURE_TAG,
    EigenGANDiscriminator,
    EigenGANGenerator,
    hinge_v1_d_loss,
    hinge_v1_g_loss,
    zero_centered_gradient_penalty,
)
from eigenforge.train_eigengan import (
    apply_determinism,
    seed_everything,
    update_ema,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data/celeba_aligned")
    parser.add_argument("--dataset-name", default="celeba", help="Recorded in the manifest only.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--z-dims", type=int, nargs="+", default=[6] * 6)
    parser.add_argument("--eps-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument(
        "--epoch-start-decay",
        type=int,
        default=160,
        help="First epoch of the linear decay to zero; >= --epochs means a constant rate.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--gp-weight", type=float, default=10.0)
    parser.add_argument("--gp-period", type=int, default=3, help="Lazy-penalty period in D steps.")
    parser.add_argument(
        "--orth-weight",
        type=float,
        default=1.0,
        help="Weight of the orthogonality penalty. 0 switches the generator to Gram-Schmidt.",
    )
    parser.add_argument("--truncation", type=float, default=1.5)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-traversal", type=int, default=5)
    parser.add_argument("--n-left-axis-point", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=1000, help="Steps; 0 disables.")
    parser.add_argument("--traversal-every", type=int, default=2500, help="Steps; 0 disables.")
    parser.add_argument("--output-dir", default="./experiments/eigengan_faithful")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to latest.pt to resume training")
    parser.add_argument(
        "--allow-unverified-resume",
        action="store_true",
        help="Permit resuming a checkpoint that has no sibling manifest.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="See train_eigengan.py.")
    return parser.parse_args()


def validate_args(args) -> None:
    if 4 * 2 ** len(args.z_dims) != args.image_size:
        raise ValueError(
            f"--image-size {args.image_size} needs {len(args.z_dims)} z-dims to be "
            f"4 * 2**len(z_dims) = {4 * 2 ** len(args.z_dims)} (official: log2(crop/4) == len(z_dims))"
        )
    if args.orth_weight < 0:
        raise ValueError("--orth-weight must be >= 0 (0 selects the Gram-Schmidt generator)")
    if args.gp_period < 1 or args.gp_weight < 0:
        raise ValueError("--gp-period must be >= 1 and --gp-weight >= 0")
    if args.epochs < 1 or args.batch_size < 1 or args.epoch_start_decay < 0:
        raise ValueError("--epochs and --batch-size must be >= 1 and --epoch-start-decay >= 0")
    if not math.isfinite(args.truncation) or args.truncation <= 0:
        # truncation=0 makes trunc_normal_ silently return all-zero latents (the
        # "mode" image for every preview); negative only blows up at the first preview.
        raise ValueError("--truncation must be finite and > 0")
    if args.n_samples < 1 or args.n_traversal < 1 or args.n_left_axis_point < 1:
        raise ValueError("--n-samples, --n-traversal and --n-left-axis-point must be >= 1")
    if args.sample_every < 0 or args.traversal_every < 0:
        raise ValueError("--sample-every and --traversal-every must be >= 0 (0 disables)")
    if not 0.0 <= args.ema_decay < 1.0:
        # 1.0 freezes the EMA at its random init: every preview and the final ema
        # checkpoint would silently come from an untrained network.
        raise ValueError("--ema-decay must be in [0, 1)")


def linear_decay_lr(initial: float, epochs: int, start_decay: int, epoch: int) -> float:
    """Official ``tl.LinearDecayLR``: constant before ``start_decay``, then linear to zero."""
    if epoch < start_decay:
        return initial
    return initial * (1 - (epoch - start_decay + 1) / (epochs - start_decay + 1))


class FlatImageDataset(Dataset):
    """Every image directly inside ``root``, sorted by name (official ``data.py``)."""

    def __init__(self, root: str, transform):
        self.paths = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def build_transform(image_size: int):
    # Official: Resize(load_size), random horizontal flip, CenterCrop(crop_size), scale
    # to [-1, 1] as x / 127.5 - 1. load_size == crop_size here, so the crop is a no-op
    # and is left out.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def infinite(loader: DataLoader):
    while True:
        yield from loader


def discriminator_loss(discriminator, generator, real, *, gp_weight: float, gp_period: int, step: int):
    """Hinge D loss; the penalty graph exists only on the steps that apply it."""
    zs, eps = generator.sample_latents(real.size(0))
    with torch.no_grad():
        fake = generator(zs, eps)
    do_gp = gp_weight > 0 and step % gp_period == 0
    real = real.detach().requires_grad_(do_gp)
    real_logits = discriminator(real)
    loss = hinge_v1_d_loss(real_logits, discriminator(fake))
    if do_gp:
        loss = loss + gp_weight * gp_period * zero_centered_gradient_penalty(real_logits, real)
    return loss


def generator_loss(discriminator, generator, batch: int, orth_weight: float):
    zs, eps = generator.sample_latents(batch)
    adversarial = hinge_v1_g_loss(discriminator(generator(zs, eps)))
    if orth_weight == 0:
        # Gram-Schmidt mode: the forward basis is already orthonormal, so the penalty on
        # the raw U is meaningless — report zero instead of a misleading large number.
        return adversarial, next(generator.parameters()).new_zeros(()), adversarial
    orthogonal = generator.orthogonal_loss()
    return adversarial, orthogonal, adversarial + orth_weight * orthogonal


@torch.no_grad()
def save_samples(ema_generator, args, output_dir: Path, step: int) -> None:
    zs, eps = ema_generator.sample_latents(args.n_samples, truncation=args.truncation)
    path = output_dir / "samples" / f"sample_{step:08d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        ema_generator(zs, eps), path, nrow=int(args.n_samples**0.5), normalize=True, value_range=(-1, 1)
    )


@torch.no_grad()
def save_traversals(ema_generator, args, output_dir: Path, step: int) -> None:
    """Every layer's dimensions, most important first; sample 0 is the all-zero "mode"."""
    zs, eps = ema_generator.sample_latents(args.n_traversal, truncation=args.truncation)
    for z in zs:
        z[0] = 0.0
    eps[0] = 0.0
    values = torch.linspace(-4.5, 4.5, args.n_left_axis_point * 2 + 1).tolist()
    folder = output_dir / "traversal" / f"step_{step:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    for layer in range(len(args.z_dims)):
        for rank, dim in enumerate(ema_generator.dims_by_importance(layer).tolist()):
            frames = [f.cpu() for f in ema_generator.traverse(zs, eps, layer, dim, values)]
            importance = ema_generator.L[layer][dim].detach().abs().item()
            save_image(
                torch.cat(frames, dim=3),
                folder / f"L{layer}_rank{rank}_dim{dim}_absL{importance:.3f}.jpg",
                nrow=1,
                padding=0,
                normalize=True,
                value_range=(-1, 1),
            )


def main():
    args = parse_args()
    enforce_safe_torch_at_startup(when="train_eigengan_faithful")
    validate_args(args)
    seed_everything(args.seed)
    if args.deterministic:
        apply_determinism()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FlatImageDataset(args.data_root, build_transform(args.image_size))
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # Official: len_dataset // (n_d + 1) iterations per epoch with n_d = 1 D step per G step.
    iters_per_epoch = len(loader) // 2
    if iters_per_epoch == 0:
        raise ValueError(
            f"dataset at {args.data_root!r} has {len(dataset)} images; need at least "
            f"2 x --batch-size ({args.batch_size}) for one epoch"
        )

    use_gram_schmidt = args.orth_weight == 0
    generator = EigenGANGenerator(args.z_dims, args.eps_dim, use_gram_schmidt=use_gram_schmidt).to(device)
    ema_generator = copy.deepcopy(generator).eval()
    for parameter in ema_generator.parameters():
        parameter.requires_grad_(False)
    discriminator = EigenGANDiscriminator(args.image_size).to(device)
    betas = (args.beta1, 0.999)
    g_optim = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=betas)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=betas)

    start_epoch = 0
    if args.resume:
        resume_state = load_checkpoint(
            args.resume,
            map_location=str(device),
            required_keys=("generator", "ema_generator", "discriminator", "g_optim", "d_optim"),
            expected_config={
                "architecture": ARCHITECTURE_TAG,
                "z_dims": list(args.z_dims),
                "eps_dim": args.eps_dim,
                "image_size": args.image_size,
                "use_gram_schmidt": use_gram_schmidt,
            },
            require_manifest=not args.allow_unverified_resume,
            fallback_to_previous=True,
        )
        payload = resume_state.payload
        generator.load_state_dict(payload["generator"])
        ema_generator.load_state_dict(payload["ema_generator"])
        discriminator.load_state_dict(payload["discriminator"])
        g_optim.load_state_dict(payload["g_optim"])
        d_optim.load_state_dict(payload["d_optim"])
        start_epoch = resume_state.epoch
        logger.info(
            "Resumed from {} at epoch {} (manifest verified={}; a fallback warning above, "
            "if any, names the file that was actually loaded)",
            args.resume,
            start_epoch,
            resume_state.manifest is not None,
        )
        if resume_state.manifest is not None:
            recorded = resume_state.manifest.get("training_config") or {}
            for key in ("batch_size", "gp_period"):
                # Not architecture (expected_config already rejects that drift): these
                # only shift the global step numbering, the lazy-penalty phase and the
                # preview triggers — legal to change, but never silently.
                if recorded.get(key) != getattr(args, key):
                    logger.warning(
                        "resume drift: {} recorded={} now={}; global step numbering and "
                        "the lazy penalty phase will shift",
                        key,
                        recorded.get(key),
                        getattr(args, key),
                    )

    if start_epoch >= args.epochs:
        raise ValueError(
            f"checkpoint is already at epoch {start_epoch} >= --epochs {args.epochs}; "
            "raise --epochs to continue training"
        )

    batches = infinite(loader)
    for epoch in range(start_epoch, args.epochs):
        lr = linear_decay_lr(args.lr, args.epochs, args.epoch_start_decay, epoch)
        for optimizer in (g_optim, d_optim):
            for group in optimizer.param_groups:
                group["lr"] = lr
        for it in range(iters_per_epoch):
            step = epoch * iters_per_epoch + it
            real = next(batches).to(device)

            d_loss = discriminator_loss(
                discriminator,
                generator,
                real,
                gp_weight=args.gp_weight,
                gp_period=args.gp_period,
                step=step,
            )
            check_finite_losses({"d_loss": d_loss}, epoch=epoch, step=step)
            d_optim.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optim.step()

            discriminator.requires_grad_(False)  # the G update needs no D gradients
            try:
                g_adv, g_orth, g_loss = generator_loss(
                    discriminator, generator, real.size(0), args.orth_weight
                )
            finally:
                # Restore before any future retry logic could train a frozen discriminator.
                discriminator.requires_grad_(True)
            check_finite_losses({"g_adv": g_adv, "g_orth": g_orth, "g_loss": g_loss}, epoch=epoch, step=step)
            g_optim.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optim.step()
            update_ema(ema_generator, generator, args.ema_decay)

            if step % 100 == 0:
                logger.info(
                    "epoch={} step={} d_loss={:.4f} g_adv={:.4f} g_orth={:.4f}",
                    epoch, step, d_loss.item(), g_adv.item(), g_orth.item(),
                )
            if (args.sample_every and step % args.sample_every == 0) or (
                args.traversal_every and step % args.traversal_every == 0
            ):
                # A finite loss can still leave NaN in the weights via the gradients,
                # and save_image's clamp does not catch NaN — without this check a
                # poisoned EMA would silently write garbage JPEGs before the epoch-end
                # parameter check ever runs.
                check_finite_parameters(
                    {"ema_generator": ema_generator},
                    epoch=epoch,
                    context="before rendering previews at",
                )
            if args.sample_every and step % args.sample_every == 0:
                save_samples(ema_generator, args, output_dir, step)
            if args.traversal_every and step % args.traversal_every == 0:
                save_traversals(ema_generator, args, output_dir, step)

        check_finite_parameters(
            {"generator": generator, "ema_generator": ema_generator, "discriminator": discriminator},
            epoch=epoch,
        )
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
            model_kind="eigengan_faithful",
            seed=args.seed,
            step=(epoch + 1) * iters_per_epoch,
            training_config={
                **vars(args),
                "architecture": ARCHITECTURE_TAG,
                "use_gram_schmidt": use_gram_schmidt,
                "torch": torch.__version__,
                "deterministic": args.deterministic,
                "device": str(device),
            },
            dataset={
                "name": args.dataset_name,
                "root": str(args.data_root),
                "num_images": len(dataset),
            },
        )


if __name__ == "__main__":
    main()
