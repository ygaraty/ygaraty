# GPU-resident DiLoCo local steps

**Result:** full-round GPU execution reduced local Fashion-MNIST training time
with matching loss and accuracy. Final-code series measured **7.2× and 8.8×**;
an earlier series measured **2.9×**. Timing on this shared desktop varied.

Weights, optimizer moments and forward/backward tensors stay on GPU across eight
local steps. The executor consumes our Tape/compiler plans, reuses buffers and
reads back once per round. Outer DiLoCo updates and evaluation are unchanged.
Unsupported plans fall back before execution; errors preserve the last checkpoint.

## Measured comparison

One Apple M1 GPU, Deno 2.9.5, Muon ConvNet (channels 8/16/16), batch 16,
eight local steps, six rounds: 768 training samples and 256 evaluation examples.
Each series uses two seeds with reversed arm order. Baseline already includes
our earlier GPU-resident Muon blocks. Times include startup and evaluation,
excluding dataset loading and initial coordinator evaluation.

| Series | Previous path, median | Full residency, median | Speedup |
|---|---:|---:|---:|
| Earlier candidate | 31.24 s | 10.85 s | 2.88× |
| Final candidate | 26.68 s | 3.72 s | 7.18× |
| Final repeat | 28.35 s | 3.22 s | 8.81× |

All pairs have identical accuracy; maximum evaluation-loss difference is
**3.05e-14**. Final losses are 1.6758 and 1.4984 for seeds 2026 and 2027.
These short runs establish execution parity, not convergence or time to a useful
target loss. Twelve timed trials are retained; no thermal/cache reset was used.

Each resident round reports zero CPU tensor tasks, nine GPU submissions and one
readback. Tracked GPU buffers peak at 12.25 MB for this configuration; this is
not total device memory usage. Optimizer-only GPU time was not measured.

## Correctness and limits

- Deno Muon and Adam, plus two Chromium contributors, pass the CPU weight-error
  gate of 5e-5 across two eight-step rounds.
- Nineteen kernel cases pass independent numerical references, including strided
  tensors, cross-entropy extremes and small optimizer denominators.
- Cancellation, nonfinite intermediate and device-loss checks pass.
- TypeScript/generated-code checks and 148 browser unit tests pass; Python
  training/protocol checks pass (33 tests, one optional browser test skipped).

Convolution retains FP32 arithmetic; other supported operations use compensated
FP32 pairs. Residency lasts one local round, not across global synchronization.
Only supported bounded plans use it. This is one physical GPU: no multi-GPU,
NVIDIA, remote time-to-loss or upstream PyTorch speedup is established.

[Reproduce and inspect every trial](results/resident-training-local/README.md).
See [execution controls and limits](../docs/native-training.md#gpu-resident-local-training).

Follow-up: [one/four NVIDIA GPUs versus MuLoCo-1](resident-training-nvidia.md).
