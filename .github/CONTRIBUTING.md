# Contributing to EigenForge (`eigenforge`)

Thank you for your interest in contributing to `eigenforge` (EigenGAN & DDPM/DDIM generative modeling suite).

## Core Engineering & Modeling Rules

Before opening a pull request, please read [docs/MODEL_CARD.md](../docs/MODEL_CARD.md) and ensure your changes preserve these invariants:

1. **Orthogonality Loss Scale**: Keep `EigenGenerator.total_orthogonality_loss()` normalized by layer count and basis count (`K^2`). Do not replace it with an unnormalized Frobenius sum unless re-tuning `ortho_weight` with full training comparisons.
2. **DDIM Timestep Grid**: Keep `_build_time_grid()` strictly descending and terminated with sentinel `-1`.
3. **Checkpoint Deserialization Safety**: Every `torch.load()` call must pass literal `weights_only=True` and route through `checkpoints.load_checkpoint()`. Do not weaken the `torch >= 2.10.0` security floor in `common/torch_guard.py` (`EIGENFORGE_ALLOW_UNSAFE_TORCH` escape hatch).
4. **Lazy R1 Regularization**: Keep `discriminator_loss_with_lazy_r1()` lazy (`--d-reg-every`, default 16) so non-regularization steps never construct the second-order autograd graph.
5. **Honest Evaluation Boundaries**: Built-in PPL/KID/IS helpers are lightweight proxy metrics unless official backends (`clean-fid`, `torch-fidelity`, `lpips`) and a fixed dataset protocol are used. CPU smoke checks must continue reporting `quality_gate="not_evaluated"`.
6. **`python -O` Safety**: Never place runtime validation or side effects inside `assert` statements.

## Verification Checklist

Run the following checks before submitting a PR:

```bash
ruff check .
python -m compileall -q .
pytest -v
python scripts/quality_gate.py --seeds 0 --train-steps 2 --mode protocol
python scripts/gpu_smoke.py --seeds 0 --steps 2
```
