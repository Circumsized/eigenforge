"""Traverse every latent dimension of every layer of an official-layout EigenGAN checkpoint.

Port of ``test_traversal_all_dims.py`` from LynnHo/EigenGAN-Tensorflow (commit fc3e1a2, MIT,
Copyright (c) 2021 Zhenliang He): for each layer, dimensions are visited by descending
``|L|``; each dimension is swept over ``linspace(-4.5, 4.5, n_traversal_point)`` while every
other latent stays fixed, and one horizontal strip per sample is written as
``{sample:04d}_{layer}-{rank}-{|L|:.3f}-{dim}.jpg``. Unlike the in-training traversal, no
sample is forced to the all-zero "mode".

Never executed. The architecture (``z_dims``, ``eps_dim``, Gram-Schmidt mode) is read from
the verified manifest next to the checkpoint — the counterpart of the official
``settings.yml`` — so a checkpoint without a manifest is refused. The EMA generator is
rendered, as in the official ``G_test``.

Example:
    PYTHONPATH=. python eigenforge/scripts/traverse_eigengan_faithful.py \
        --checkpoint experiments/eigengan_faithful/latest.pt
"""
import argparse
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eigenforge.checkpoints import load_checkpoint
from eigenforge.common.logger import logger
from eigenforge.common.torch_guard import enforce_safe_torch_at_startup
from eigenforge.models.eigengan_faithful import ARCHITECTURE_TAG, EigenGANGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-layout EigenGAN: traverse all dimensions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="./experiments/eigengan_faithful/traversal_all_dims")
    parser.add_argument("--n-traversal", type=int, default=100, help="Samples per strip set.")
    parser.add_argument("--n-traversal-point", type=int, default=17)
    parser.add_argument("--truncation", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not math.isfinite(args.truncation) or args.truncation <= 0:
        parser.error("--truncation must be finite and > 0")
    if args.n_traversal < 1 or args.n_traversal_point < 1:
        parser.error("--n-traversal and --n-traversal-point must be >= 1")
    # An externally supplied checkpoint is the widest untrusted-input surface; after
    # parse_args so --help works even on a torch below the CVE floor.
    enforce_safe_torch_at_startup(when="traverse_eigengan_faithful")

    device = torch.device(args.device)
    state = load_checkpoint(
        args.checkpoint,
        map_location=str(device),
        required_keys=("ema_generator",),
        expected_config={"architecture": ARCHITECTURE_TAG},
        require_manifest=True,
        fallback_to_previous=True,
    )
    config = state.manifest["training_config"]
    generator = EigenGANGenerator(
        config["z_dims"], config["eps_dim"], use_gram_schmidt=bool(config["use_gram_schmidt"])
    ).to(device)
    generator.load_state_dict(state.payload["ema_generator"])
    generator.eval()
    logger.info("Loaded {} (z_dims={}, eps_dim={})", args.checkpoint, config["z_dims"], config["eps_dim"])

    torch.manual_seed(args.seed)
    zs, eps = generator.sample_latents(args.n_traversal, truncation=args.truncation)
    values = torch.linspace(-4.5, 4.5, args.n_traversal_point).tolist()
    folder = Path(args.output_dir) / (
        f"traversal_{args.n_traversal_point}_{args.truncation:.2f}"
        f"_n{args.n_traversal}_seed{args.seed}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    image = generator.image_size
    # One reused strip buffer instead of a frames list + torch.cat: halves the host
    # memory peak (the strip itself is irreducible output).
    strip_buffer = torch.empty(args.n_traversal, 3, image, image * len(values))
    for layer in range(len(generator.z_dims)):
        for rank, dim in enumerate(generator.dims_by_importance(layer).tolist()):
            for offset, frame in enumerate(generator.traverse(zs, eps, layer, dim, values)):
                strip_buffer[..., offset * image : (offset + 1) * image] = frame.cpu()
            importance = generator.L[layer][dim].detach().abs().item()
            for index, strip in enumerate(strip_buffer):
                save_image(
                    strip,
                    folder / f"{index:04d}_{layer}-{rank}-{importance:.3f}-{dim}.jpg",
                    normalize=True,
                    value_range=(-1, 1),
                )
    print(f"Traversal strips saved -> {folder}")


if __name__ == "__main__":
    main()
