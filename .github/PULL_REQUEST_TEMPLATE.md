## Summary
- 

## Validation Checklist
- [ ] `pytest -v` passes
- [ ] `ruff check .` passes
- [ ] `python -m compileall -q .` passes
- [ ] `python scripts/quality_gate.py --seeds 0 --train-steps 2 --mode protocol` passes
- [ ] `python scripts/gpu_smoke.py --seeds 0 --steps 2` passes
- [ ] All `torch.load()` calls pass `weights_only=True` via `checkpoints.load_checkpoint`
