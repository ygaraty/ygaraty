# Resident Muon on NVIDIA

**2026-11-24 · RTX 3060, 12 GiB · Deno 2.9.5 / Vulkan · one worker**

Resident Newton–Schulz execution made short Fashion-MNIST training **1.67× faster**:
median **68.97 → 41.22 seconds**, with matching loss and accuracy.

| Seed | Original tasks | Resident execution | Final loss, both | Accuracy, both |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 68.80 s | 41.66 s | 1.675814 | 42.58% |
| 2027 | 69.14 s | 40.79 s | 1.498360 | 39.45% |

- Same ConvNet (channels 8/16/16), 768 training examples, batch 16, eight local
  steps and six outer rounds per trial. Evaluation: 256 held-out images every
  two rounds. Arm order reversed for the second seed; no thermal/cache reset.
- Complete Muon updates for the three convolution matrix shapes were
  **6.7–7.7× faster** over seven paired, warmed repetitions. Maximum optimizer
  error across seven edge cases: **2.3e-15**. Full-training CPU-oracle weight
  error: **9.5e-7**, below the **5e-5** gate.
- Maximum final-loss difference: **7.6e-15**. All four timed trials passed.
  Timing includes contributor startup, training and evaluation; initial
  coordinator evaluation and dataset download are excluded.
- One physical GPU was verified against Vulkan; driver **580.178.04**. This
  short run measures execution cost, not convergence quality or multi-GPU scaling.
  The tested snapshot is pinned in `source.json`; implementation and evidence
  were later published in [450b147](https://github.com/ygaraty/ygaraty/commit/450b147eb37a74a0bf0f98154f0efbf46e7527dd). This is experimental, not a release.

**Provider-reported total cost: $0.013**, including setup failures.
All experiment resources were destroyed after testing.

[Results and reproduction](results/muon-resident-nvidia/README.md) ·
[Local M1 comparison](muon-resident-local.md)
