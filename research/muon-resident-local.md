# Resident Muon: local execution speed

**2026-11-24 · Apple M1 · Deno/WebGPU · one worker**

Keeping Newton–Schulz intermediates on the GPU made short Fashion-MNIST training
**2.14× faster**, with matching final loss and accuracy.

| Seed | Original task execution | Resident execution | Final loss, both |
| --- | ---: | ---: | ---: |
| 2026 | 61.26 s | 28.68 s | 1.675814 |
| 2027 | 61.72 s | 28.90 s | 1.498360 |

Each trial used 768 examples, batch 16, eight local steps, six outer rounds,
channels 8/16/16, and 256 held-out evaluation examples. Trial order alternated
by seed. Maximum final-loss difference: **7.4e-15**. All four trials passed.
This short comparison measures execution cost, not convergence quality.

The three convolution matrix shapes showed **7.2–9.0×** faster complete Muon
updates in seven paired, warmed repetitions. Maximum numerical error across
seven edge cases, including tall matrices and zero gradients: **2.3e-15**.

Full-training CPU-oracle checks passed with one Deno worker and two Chromium
contributors; maximum weight error **9.5e-7**, below the **5e-5** gate.
Both browser contributors shared one GPU.

The compiler retains portable Tape operations. Each supported matrix now uses
one GPU submission/readback for its Newton–Schulz block instead of 15 matrix
readbacks: **45 → 3 per step** for this ConvNet. Compensated arithmetic is retained.
Momentum, normalization, final updates and forward/backward execution remain
unchanged. Tiled graphs use the original path; buffers are not retained across steps.

This local run used no cloud resources. A [separate NVIDIA run](muon-resident-nvidia.md)
now validates that backend. See
[raw results and reproduction commands](results/muon-resident-local/README.md).
