# WebGPU precision fix: compensated multiplication and accumulation

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | HTTP + browser (compensated WebGPU) |
| **Result** | Two-float32 compensated matmul restored 511/511 checks with unchanged tolerances; max Adam-update error 9.08e-12. |
| **Evidence** | `runs/gpu-precision-fix.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

The Transformer WebGPU recipe now defaults to `gpu_arithmetic="compensated"`.
`"float32"` remains available via the creation API for reproducible diagnostics;
MLP/LSTM's existing GPU behavior is unchanged. Compensated arithmetic uses **two
float32 components per value**, not native float64 GPU instructions. Each input
is split into a high part and its residual. The shader uses FMA product residuals,
TwoSum accumulation and high/low outputs; JavaScript reconstructs the output in
float64. Matrix products are still computed entirely on the GPU.

The initial unguarded compensation shader lost its residuals on this GPU. The
fixed shader places a float→integer→float round-trip with a runtime-zero XOR at
rounding boundaries, preventing algebraic optimization across those boundaries.
The fourth dimension-uniform word must remain zero; the SDK always supplies zero.
Real-browser cancellation/input-rounding/product-rounding regressions verify this
behavior. This is approximately double-float precision within float32's exponent
range, not a claim of universal IEEE float64 equivalence. Other GPU drivers still
need the real-GPU regression and end-to-end tests.

The diagnosis used a fresh reproduction of the original kernel, recording actual
and reference gradients at the largest Adam differences. At
`decoder.layers.0.self_attn.in_proj_weight[72,172]`, the reference gradient was
-5.004136e-10 but the float32 path returned +2.177330e-9. With the **unchanged**
learning rate 1e-4 and epsilon 1e-8, the reference first-step Adam delta was
+4.765656e-6 and the GPU delta was -1.788019e-5. Replaying Adam from the actual
GPU gradients matched all updated weights to 2.78e-17. The optimizer was applying
the perturbed gradients correctly; the precision loss occurred upstream.

The corrected full-model browser run (`72a313ff39c6484fa280117b21c26ae5`) completed
all 1,344 GPU matmuls and all 6,923 browser computations with zero retries or
rejections. **511/511 comparisons passed**, including all original strict
float64 comparisons. No tolerance, optimizer, checkpoint, tokenizer, fixture or
label-smoothing changes were made.

| Maximum absolute error vs PyTorch | Original GPU | Compensated GPU |
| --- | ---: | ---: |
| Logits | 8.20e-6 | 3.38e-13 |
| Loss | 5.52e-8 | 2.58e-14 |
| Gradients | 8.20e-7 | 8.82e-13 |
| Adam updates | 2.26e-5 | 9.08e-12 |

Audit: `runs/gpu-precision-fix.json`. `ygaraty/experiments/verify_gpu_precision_fix.py` asserts identical
reference fingerprints, optimizer settings and per-tensor tolerances, checks all
GPU matmuls ran, and summarizes before/after errors. Diagnostic gradients are
written temporarily per parameter and deleted after its Adam comparison; only
small diagnostic examples remain in the report.

Focused GPU regression: serve `ygaraty/webapp/static/` on localhost and open
`gpu_precision_test.html`, then click **Run GPU precision tests**. It exercises
the actual shaders and alternates plain/compensated pipelines, with no CPU
fallback. Result: `runs/gpu-precision-regression.json`. A compensated CPU sum is
used for the cancellation-test oracle so the reference itself does not lose the
small terms. The full Transformer oracle remains the original PyTorch result.
