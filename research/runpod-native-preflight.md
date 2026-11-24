# RunPod native training preflight — November 20, 2026

**Outcome:** one RTX 4090 passed native training parity. The requested 1-versus-4
comparison could not start because RunPod rejected additional matching Pods for
capacity. All experiment Pods were deleted and inventory cleanup was verified.

The public wheel from `feature/public-diloco-training` was installed on Ubuntu
24.04.5 with Python 3.12.3, NumPy 2.5.5 and Deno 2.9.5. Runtime source hash:
`14f4648635ae5642fff90a60c122cef95f7a51dc51620207e6d714eed43d200e`.
The base commit and uncommitted-source status are in the
[attempt record](results/runpod-native-preflight/attempts.json).

The first Vulkan probe selected llvmpipe despite `nvidia-smi` identifying the
RTX 4090. The training worker correctly rejected that software adapter. Installing
the missing EGL/GL loader libraries enabled the hardware adapter. The successful
run reported NVIDIA driver 595.112.07 and `isFallbackAdapter: false`.

Three shared Adam updates used the residual ConvNet with channels 8/16/16,
Fashion-MNIST, global batch 40 and seed 2026. Each update was checked against the
unsplit Python reference. The largest absolute weight difference was
**7.41e-7**, below the `2e-5` gate. Wall time was **5.17 seconds**, including
reference checks and final evaluation; setup and initial evaluation were excluded.
Only 16 held-out examples were evaluated. This is numerical smoke evidence,
not convergence, multi-GPU scaling or a performance benchmark.

The second same-datacenter RTX 4090 Pod was rejected for capacity. Attempts in
two cheaper RTX PRO 4500 pools were also rejected. A shorter A100 fallback placed
one Pod but could not place a second; that Pod was released without training.
No four-worker training result exists, and no GPU slices were substituted for
physical GPUs.

Estimated compute cost was **$0.16**, excluding storage and tax. Billing records
had not yet included these Pods when checked. The authorized target was $7.12,
with a $15 hard cap. An independent watchdog was armed before provisioning;
explicit deletion ended both rentals early. No persistent or network volumes
were created. A worker's SIGTERM cleanup did not finish reliably, so final
cleanup was verified through Pod deletion rather than assumed from process exit.

Evidence: [training report](results/runpod-native-preflight/report.json),
[final weights](results/runpod-native-preflight/report.npz),
[cost basis](results/runpod-native-preflight/cost.json), and
[verified cleanup](results/runpod-native-preflight/cleanup.json).
