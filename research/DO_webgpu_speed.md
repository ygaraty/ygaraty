# WebGPU worker speed: closing the gap with WebLLM and native

Status: planned 2026-11-12. Experiments WS-30 to WS-38 and DO-67, in the order below.

## Where we are (experiments 113–117, DO-61)

Kolibri-1 on one A100 80 GB, one Chrome page, one stage of 50 layers: **47 ms per token in the harness** (about 21 tokens per second), of which **28 ms is GPU work** and about 19 ms host work (JavaScript encoding of 1,238 dispatches 7.6 ms, plan expansion 3.8 ms, submit, wait and the token's read-back). The GPU time by operation (experiment 115, short context): Q6_K matrix-vector products 29%, attention 24% (0.21 ms per layer at about 40 positions), RMS norms 13%, routed experts 10%. Kolibri reads about 3 GB of weights per token; at the A100's 2 TB/s the bandwidth floor is about 1.5 ms. Through production the token loop no longer waits for the coordinator (DO-61 part 1).

What the kernels do not use yet: WebGPU `subgroups` (reductions in matrix-vector products, norms, attention and argmax go through workgroup memory and barriers) and `shader-f16` (f16 data is unpacked from u32 words; arithmetic is f32). Each token is read back to JavaScript before the next is encoded, so the GPU idles during host work.

## The reference

WebLLM (MLC-LLM and Apache TVM: kernels compiled and tuned ahead of time, kernel fusion, GEMM tiling, paged KV cache, CPU-side work in WASM) reports up to about 80% of native MLC-LLM decode throughput on the same device: Llama-3.1-8B 41.1 against 57.7 tokens per second (71.2%), Phi-3.5-mini 71.1 against 89.3 (79.6%), on an Apple M3 Max in Chrome Canary 133 (WebLLM v0.2.75; arXiv 2412.15803v2, Table 1). It publishes no NVIDIA or AMD numbers, so WS-30 and DO-67 measure them on our machines.

**Metric that compares different models:** effective bandwidth = bytes of weights read per decode token × tokens per second, as a share of the device's peak. Native llama.cpp and MLC typically reach 50–80% of peak on dense 8B Q4 decode; that, not a model's raw tokens per second, is the bar.

## First measurement (experiment 118, WS-30, 2026-11-12)

On the Radeon 680M with Qwen3.5-9B: llama.cpp Vulkan 12.3 tokens per second (62.7 GB/s), WebLLM 8.9 (72% of native, 39.8 GB/s), ours 8.0 (65%, 40.8 GB/s). Per byte of weights we are level with WebLLM; its lead is its smaller 4-bit format. Both browsers stop at about two thirds of native on this GPU. On big GPUs the gap is host time (experiment 117), so WS-32 and WS-33 come first for the A100; the kernel experiments (WS-34, WS-35) close the last third on both. A lighter weight format (Q4_K everywhere instead of Q4_K_M's Q6_K tensors) is a separate lever worth an experiment.

## Progress (2026-11-12)

- Experiment 120 (WS-32): the GPU-resident loop hides the host work on the A100 (Kolibri 73 → 35 ms per token at 64 positions); harness only so far.
- Experiment 121 (WS-34): the coalesced kernel puts the matrix-vector products at the 680M's roof; Qwen3.5-9B 10.4 tokens per second, 85% of llama.cpp. Opt-in so far.
- Experiment 122 (WS-35): 64-key attention splits take Kolibri's attention from 94 to 14 ms per token at 1k positions; with the resident loop Kolibri-1 decodes at about 33 tokens per second on one A100 in the harness. Failed gate on a long-prompt accuracy gap that the old splits share.

## Targets

| | Now (Kolibri, A100) | Target |
|---|---|---|
| Host time not overlapped with GPU | about 19 ms | under 1 ms (WS-32, WS-33) |
| GPU time per token | 28 ms | under 8 ms (WS-34 to WS-37) |
| Effective bandwidth | about 6% of peak (3 GB / 47 ms) | at least 40% of peak, and at least WebLLM's share on the same device (WS-30, DO-67) |

## Order

1. **WS-30**: the gap, measured on the workstation: WebLLM, ours and llama.cpp (Vulkan) on comparable 4-bit 8B-class models; effective bandwidth for each. Defines the bar.
2. **WS-31**: steady roofline bench (fixes experiment 115: a fresh slot per variant, positions 64, 1k, 4k, GB/s and share of peak per operation). Every later experiment reads its numbers from it.
3. **WS-32**: GPU-resident decode loop: the token stays on the GPU (argmax writes it, the next embedding reads it); token n+1 is submitted before token n is read back. Hides host time. Bitwise identical.
4. **WS-33**: lean encoding: a frozen per-token dispatch list (no plan expansion), position uniforms by dynamic offset, fewer pipeline switches. Host time per token under 3 ms. Bitwise identical.
   Done 2026-11-13; see experiment 145: caching shader sources cut host ms per step 10% on Kolibri (Rejected; kept: −24–30% on Qwen3-Coder and Qwen3.5); JavaScript is about 6 µs per dispatch on the workstation, so the frozen list waits for an A100 host breakdown.
5. **WS-34**: subgroup matrix-vector kernels (and `shader-f16` where it helps), with K-quant blocks repacked at load into a GPU-friendly layout. Q4_K, Q5_K, Q6_K, Q8_0 at at least 60% of peak bandwidth on the 680M. Within tolerance, tokens equal.
6. **WS-35**: flash-decoding attention: positions split across workgroups, subgroup reductions, a fixed-order combine. Short context under 0.03 ms per layer on the A100 equivalent; linear at 4k. Within tolerance, tokens equal.
7. **WS-36**: fusion round 2: RoPE with the cache store, SwiGLU in the gate/up product's epilogue, residual adds in the output products' epilogues, router top-k in one kernel. Dispatches per token −40%. Bitwise identical where the arithmetic order is kept.
   Done 2026-11-13; see experiment 135: products of the same input grouped (518 → 366 dispatches per token, tokens identical, step unchanged on the 680M); RoPE + store and router top-k not done.
8. **WS-37**: MoE expert path: the top-k experts' products in one dispatch with gathered rows (indirect dispatch), no per-expert host work. Kolibri.
9. **WS-38**: prefill GEMM with subgroups (and Chrome's subgroup-matrix where available): first-token latency.
10. **DO-67**: one rented A100: the full stack against the 113–117 baseline, with WebLLM and llama.cpp (CUDA) on the same machine. The answer to "the best we can do today".
   Done 2026-11-12; see experiment 123: Qwen3.5-9B decode ours 30 tok/s, WebLLM 23; WebGPU copy roof 725 GB/s on the A100.

Every change that keeps the arithmetic is gated bitwise (tokens and logits); kernel rewrites that change summation order are gated against the NumPy reference (all tokens equal, logits within 2e-3 of the largest, as experiment 113) and WS-27's repeatability. Everything except the embedding and head applies to layer-only workers.


Ownership note (2026-11-15): public WebGPU projection; full original comparison retained in the private archive. No measurements were rerun.
