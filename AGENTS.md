# EigenForge Agent Instructions

## Purpose and layout

This repository (checkout folder name may differ) is a self-contained generative modeling repository using the standard PyPA `src`-layout (`src/eigenforge/`), implementing EigenGAN and DDPM/DDIM face-generation models, evaluation protocols, checkpoint safety primitives, and CLI entry points.

- `src/eigenforge/models/`: `eigengan.py` (simplified variant with normalized Gram-MSE constraint and sign-stabilized QR), `eigengan_faithful.py` (official-layout EigenGAN with unnormalized `0.5·Σ` orthogonal term), `diffusion_scheduler.py`, `unet_diffusion.py`
- `src/eigenforge/evaluation/`: `evaluator.py` (proxy metrics PSNR/SSIM/KID/IS/PPL and latent subspace traversal grids), `official_metrics.py` (optional `clean-fid` / `torch-fidelity` / `lpips` adapters and three-seed aggregation), `entropy_coefficient.py` (paper Table 1 metric), `linear_case_mle.py` (paper Appendix A closed-form MLE), `toy_basis_similarity.py` (paper Tables 2-3 experiment)
- `src/eigenforge/common/`: `config.py` (`AppConfig`), `logger.py` (Loguru logging), `fsutil.py` (durable atomic file replacement), `torch_guard.py` (runtime PyTorch CVE floor guard)
- `src/eigenforge/train_eigengan.py`, `train_diffusion.py`, `train_eigengan_faithful.py`: training entry points
- `src/eigenforge/checkpoints.py`, `artifacts.py`, `safetensors_io.py`, `smoke.py`: crash-consistent checkpoint commit/load protocol, provenance manifests, safetensors export/load, and smoke protocols
- `scripts/`: `gpu_smoke.py`, `quality_gate.py`, `traverse_eigen_dims.py`, `traverse_eigengan_faithful.py`
- `tests/`: unit and contract regression tests (`conftest.py` + `test_*.py`)
- `docs/`: structured documentation suite (`INDEX.md`, `MODEL_CARD.md`, `ARCHITECTURE_AND_MATHEMATICS.md`, `EVALUATION_AND_QUALITY_GATES.md`, `CHECKPOINT_AND_SECURITY_SPEC.md`, `CLI_AND_CONFIGURATION_REFERENCE.md`)
- `configs/eigenforge.yaml`: read-only reference mirror; runtime code does not load it
- `.github/`: CI workflow (`workflows/ci.yml`), `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, PR and Issue templates

## Validation commands

Run from the repository root:

```bash
pytest -v
ruff check .
python -m compileall -q .
python scripts/quality_gate.py --seeds 0 --train-steps 2 --mode protocol
python scripts/gpu_smoke.py --seeds 0 --steps 2
```

GPU smoke and full training require suitable hardware/data; without CUDA, report `not_evaluated` rather than inventing quality results.

## Non-negotiable model and security constraints

- Preserve `total_orthogonality_loss()`'s normalized Gram-MSE scale; do not replace it with an unnormalized Frobenius sum without retraining and comparison.
- DDIM timestep grids must be strictly descending and end with sentinel `-1`.
- Every `torch.load()` must pass `weights_only=True` and route through `checkpoints.load_checkpoint`; torch-based entry points require `torch >= 2.10.0` (CVE-2025-32434, fixed 2.6.0; CVE-2026-24747, fixed 2.10.0).
- R1 regularization is lazy (`--d-reg-every`, default 16); non-regularization steps must not build the R1/autograd graph.
- Use the official transforms bundled with Inception weights; do not substitute `transform_input=True`.
- PPL/KID/IS are lightweight proxy metrics unless an official backend and fixed dataset/protocol are explicitly enabled; passing tests or smoke checks is not a quality claim.
- Do not put side effects or runtime validation in `assert`; code must remain correct under `python -O`.
- Keep secrets and machine-specific paths in `AppConfig`/`.env`; never hard-code credentials or local paths.

Before changing model behavior or evaluation, read `docs/MODEL_CARD.md` and compare the affected model files with `src/eigenforge/evaluation/evaluator.py`. Preserve artifact manifests and atomic write semantics when modifying checkpoint or output handling.
