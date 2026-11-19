"""Traverse EigenGAN latent axes and export comparison grids.

Example:
    python eigenforge/scripts/traverse_eigen_dims.py --checkpoint experiments/eigengan/latest.pt \
        --output experiments/demo/traversal.png
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eigenforge.checkpoints import load_checkpoint, select_generator_state
from eigenforge.common.torch_guard import enforce_safe_torch_at_startup
from eigenforge.evaluation.evaluator import GenAIEvaluator
from eigenforge.models.eigengan import EigenGenerator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EigenGAN latent-axis traversal")
    parser.add_argument("--checkpoint", default=None, help="Optional generator checkpoint")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-basis", type=int, default=6)
    parser.add_argument("--layer", type=int, default=0, help="Generator layer to probe")
    parser.add_argument("--basis", type=int, default=0, help="Eigen-dimension to traverse")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="experiments/demo/traversal.png")
    parser.add_argument(
        "--allow-unverified-resume",
        action="store_true",
        help="Allow loading a checkpoint that lacks a sibling sha256 manifest.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    # This script loads an externally supplied checkpoint: the widest
    # untrusted-input surface of the torch.load call sites. After
    # parse_args so --help works even on a torch below the CVE floor.
    enforce_safe_torch_at_startup(when="traverse_eigen_dims")

    device = torch.device(args.device)
    generator = EigenGenerator(img_size=args.image_size, latent_dim=args.latent_dim, n_basis=args.n_basis).to(device)
    if args.checkpoint:
        # Same verified path as resume: sha256 against the sibling manifest when
        # one exists, previous-generation fallback after an interrupted save, and
        # the architecture recorded in the manifest must match these flags.
        # require_epoch=False because a bare generator state dict is accepted here.
        state = load_checkpoint(
            args.checkpoint,
            map_location=str(device),
            expected_config={
                "z_dim": args.latent_dim,
                "image_size": args.image_size,
                "n_basis": args.n_basis,
            },
            require_epoch=False,
            require_manifest=not args.allow_unverified_resume,
            fallback_to_previous=True,
        )
        generator.load_state_dict(select_generator_state(state.payload, source=args.checkpoint))
        print(f"Loaded checkpoint: {args.checkpoint} (manifest verified={state.manifest is not None})")

    evaluator = GenAIEvaluator(device=str(device))
    grid = evaluator.generate_subspace_traversal_grid(
        generator,
        layer_idx=args.layer,
        basis_idx=args.basis,
        output_path=args.output,
    )
    print(f"Traversal grid saved -> {args.output} (shape={tuple(grid.shape)})")


if __name__ == "__main__":
    main()
