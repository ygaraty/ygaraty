# Browser precision comparison: WebGPU versus float64

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | HTTP + browser (float64 vs WebGPU float32) |
| **Result** | Plain float32 WebGPU failed 48/511 tensor checks: 234 Adam updates exceeded budget. Float64 passed all 511. |
| **Evidence** | `runs/browser-precision-comparison.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

The Transformer form lets you choose **JavaScript float64** or **compensated
WebGPU matmul**. Jobs dispatch work immediately to the first contributor.
Only the dedicated two-worker benchmark harness synchronizes worker startup.
The WebGPU recipe requires actual GPU matmuls: it does not silently fall back to
CPU. Other operations, including LayerNorm, softmax and Adam, still use JS
float64. Both modes use the identical checkpoint, minibatch, tile 512, dropout-off
policy, fresh Adam settings and PyTorch float64 oracle. No learning-rate or epsilon
changes are made to improve the comparison.

The server still checks each task against NumPy. Separate end-to-end checks
record maximum absolute error, relative L2 error, elementwise violations and the
strict float64 comparison for all 511 tensors. GPU comparison continues after
an error-budget violation so every gradient and optimizer tensor gets measured;
completion is distinct from passing verification. The dashboard explicitly shows
**OUTSIDE ERROR BUDGET** and allows downloading the failed comparison report.

GPU thresholds, fixed before the first experiment, use
`abs(actual-reference) <= atol + rtol*abs(reference)`:

| Tensor | rtol | atol |
| --- | ---: | ---: |
| Logits | 2e-4 | 2e-4 |
| Loss | 2e-5 | 2e-5 |
| Gradient | 2e-3 | 2e-6 |
| Adam first moment | 2e-3 | 2e-7 |
| Adam second moment | 1e-2 | 2e-8 |
| Adam **weight change** | 1e-2 | 1e-6 |

These are engineering comparison criteria, not a proof of convergence. The Adam
check compares `updated_weight - initial_weight` on both sides; a large existing
weight cannot hide an inaccurate update. Float64 continues to use rtol 2e-7,
atol 2e-9. Reports include each tensor's exact criterion and violations.

`active_seconds` starts at the first assignment after the contributor barrier,
excluding job creation and waiting for the second worker. It includes scheduling,
transport and verification. Worker compute time is summed across contributors
and includes GPU upload/dispatch/readback; it is not isolated shader time.
Run modes sequentially on the same machine; repeated randomized trials would be
needed for a reliable speed benchmark.

```sh
ygaraty experiment compare_browser_precision \
  --float64 runs/http-transformer-CPU_ID/verification.json \
  --webgpu runs/http-transformer-GPU_ID/verification.json \
  --output runs/browser-precision-comparison.json
```

The comparison tool rejects mismatched checkpoint/tokenizer/fixture hashes,
model/optimizer settings or worker counts, and checks that GPU execution occurred.


2026-10-22 measured comparison (same 19.7M model, 2 real browser workers,
11,092 total operations, 6,923 browser computations each):

| Measurement | Float64 | WebGPU repeat |
| --- | ---: | ---: |
| Active wall time | 129.47 s | 141.77 s |
| Summed worker compute incl. GPU transfers | 21.60 s | 23.06 s |
| Actual GPU matmuls | 0 | 1,344 |
| Maximum logit error vs PyTorch | 2.58e-14 | 8.20e-6 |
| Maximum gradient error | 8.89e-13 | 8.20e-7 |
| Maximum Adam weight-change error | 9.01e-12 | 2.26e-5 |
| Failed tensor comparisons | 0 / 511 | 48 / 511 |

Loss, logits, every gradient and both Adam moments passed the preset GPU budgets.
**234 scalar updates out of 19,661,824 exceeded the Adam-update budget**, across
48 parameter tensors. Aggregate update relative L2 error was 1.35e-4, but that
does not override the individual failures. Adam can amplify gradient roundoff
near zero (with the original epsilon 1e-8); the float64 default remains verified,
and the GPU comparison is explicitly marked outside budget.

Both runs had zero retries/rejections and transferred approximately 5.32 GB of
application payloads. An exploratory GPU trial took 117.72 s and reproduced the
exact same numerical checks. It overlapped a regression test and was excluded
from the primary speed comparison; the clean repeat ran after the float64 trial
with no concurrent test workload. These few measurements do not establish a
reliable speedup. Two browser contributors share one local GPU, and the many
small tasks include HTTP/storage/verification overhead.

Full results: `runs/browser-precision-comparison.json`. CPU job:
`fb552357ae7a448489e7a7b1c2854ed8`; GPU repeat:
`1604e7e333744e13a9f09f17fc4ada05`. Each run has its independent report and event
journal under `runs/http-transformer-ID/`. The GPU repeat remains inspectable
in the webapp on port 8093. Existing services on 8090 and 8092 were preserved.

After the precision-mode changes, the full regression suite passed 29 tests;
24 external Redis/PostgreSQL tests were skipped. The suite includes strict CPU
routing, required-WebGPU behavior without silent fallback, and complete Adam-delta
checks through the JavaScript HTTP SDK. The former two-worker start barrier has
since been removed; regression tests now check immediate single-worker dispatch.
