# PyTorch parity baseline across seven executors, including real WebGPU

| | |
|---|---|
| **Date** | 2026-10-24 |
| **Path** | In-process NumPy, ZeroMQ workers, real HTTP server with Node SDK workers, headless Chrome WebGPU (Apple M1, Metal) |
| **Result** | 37/38 runs passed. Plain float32 WebGPU fails the Transformer's Adam update (attention `in_proj_bias`, 73× over budget); compensated WebGPU passes. |
| **Evidence** | `runs/parity/baseline/` (local, not in git) |

The parity harness (historical internal note; not part of this repository) (`ygaraty parity`) runs registered tiny models through each executor for several Adam updates and compares logits, loss, every gradient, Adam update deltas, weights and both moments against PyTorch float64.

- **Local and ZeroMQ:** 4 cases (`lstm-tiny-corpus`, `lstm-edge-2layer`, `lstm-edge-1layer`, `transformer-tiny`) × 4 executors (`untiled`, `tiled`, `grouped`, `zmq`) × free and anchored modes = 32 runs, all passed. Reruns are bit-identical.
- **HTTP and WebGPU:** 6 free-mode runs on `lstm-tiny-corpus` and `transformer-tiny`. Every matmul ran on the GPU (e.g. 3,132/3,132 for the Transformer).

| Case | `http` (JS float64) | `webgpu` (float32 matmul) | `webgpu-compensated` |
|---|---|---|---|
| `lstm-tiny-corpus` | pass | pass | pass |
| `transformer-tiny` | pass | **fail**: attention `in_proj_bias` Adam delta 73× over budget | pass (worst delta ratio 3.6e-6) |

This reproduces the [WebGPU precision finding](014-webgpu-vs-float64-precision.md) and its [fix](015-webgpu-compensated-precision-fix.md) on tiny models in an automated, repeatable run.

Two further observations from the same work:

- With `lr=1e-3`, the documented float64 tolerance (`rtol=2e-7, atol=2e-9`) is dominated by `atol` for Adam deltas of about 1e-3, so the effective relative tolerance is about 2e-6. An injected 1e-6 relative optimizer error passes, but raises the worst ratio from about 3e-8 to 0.45, which the baseline diff flags.
- Grouped and tiled plans produce bit-identical results, so task grouping is purely a scheduling choice.

## Reproduce

```bash
ygaraty parity baseline                                   # local + ZeroMQ, ~70 s
ygaraty parity baseline --add --case lstm-tiny-corpus transformer-tiny \
  --executor http webgpu webgpu-compensated --mode free --require   # ~5 min, needs Node and Chrome
ygaraty parity check
```
