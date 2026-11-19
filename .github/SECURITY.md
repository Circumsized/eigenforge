# Security Policy

## Scope

`eigenforge` is a generative modeling research and teaching codebase (EigenGAN and DDPM/DDIM). The primary security surface is **model checkpoint deserialization** and **artifact integrity verification**.

## Checkpoint & Deserialization Security

- All PyTorch checkpoints are loaded exclusively through `checkpoints.load_checkpoint()` with literal `weights_only=True` or via `safetensors_io.load_safetensors()`.
- Entry points enforce **`torch >= 2.10.0`** via `common/torch_guard.py` because `weights_only=True` in earlier versions was vulnerable to arbitrary code execution / memory corruption (CVE-2025-32434 fixed in 2.6.0; CVE-2026-24747 fixed in 2.10.0).
- Checkpoints are verified against their sibling `.manifest.json` SHA-256 digest in a single read pass before unpickling (preventing TOCTOU file-swap attacks).
- Never load untrusted `.pt` / `.pth` checkpoints with `EIGENFORGE_ALLOW_UNSAFE_TORCH=1` or `--allow-unverified-resume`.

## Reporting a Vulnerability

Please do not open a public issue for unpatched security vulnerabilities. Report vulnerabilities privately through the repository owner's GitHub Private Vulnerability Reporting channel with a minimal reproduction and affected commit SHA.
