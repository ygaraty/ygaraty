# Caching shader sources cuts host encoding only 10% on Kolibri (24–30% on Qwen3-Coder and Qwen3.5): JavaScript is not most of the host time

| | |
|---|---|
| **Conclusion** | Rejected: the shader-source cache cuts Kolibri's host ms per step 10% (1.45 → 1.30), tokens identical (measured, 680M) |
| **Status** | executed |
| **ID** | WS-33 (experiment 145) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-12; run 2026-11-13 |
| **Path** | The plan executor builds each step's dispatch list once per shape (pipeline, bind group, workgroups) and replays it; per-step values (position, lengths) in one uniform buffer read by dynamic offset; dispatches sorted to cut pipeline switches where order allows. Arms: current vs frozen; tiny Kolibri and Qwen3.5-9B on the 680M |
| **Result** | 680M, N (no cache) against M (shader generators memoized), host ms per step = expand + encode, median of the base variants: Kolibri-1 first 8 layers 1.45 → 1.30 (**−10%**, 206 dispatches), Qwen3-Coder first 12 layers 1.85 → 1.40 (−24%, 222), Qwen3.5-9B 3.85 → 2.70 (−30%, 366); tokens identical in every run of every model. Gate (Kolibri-8, at least 50%; under 20% is Rejected): **Rejected**. The cache is kept (same WGSL text, bit-identical outputs, never slower, 1.15 ms less host per Qwen3.5 token); `--no-wgsl-memo` switches it off for measurements. On this CPU the JavaScript host work is about 6 µs per dispatch (about 8 ms per full Kolibri token), so the A100 hosts' host-bound MoE steps need a measurement on those machines before the frozen dispatch list is worth building. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/wgsl-memo-680m-2026-11-13.json`; raw `runs/experiments/WS-33/omarchy-20261113-222042/` and the baseline `runs/experiments/WS-33/omarchy-20261113-2148-baseline/`; code in this commit (`wgslMemo` in webgpu_kernels.mts, used by every shader generator in webgpu_kernels, kolibri_gpu, qwen35_gpu and dense_gpu) |

## Question

After the caches of experiment 116, encoding is 7.6 ms and plan expansion 3.8 ms per Kolibri token. WebGPU has no reusable command buffers, so encoding must be as close to a tight loop of setPipeline/setBindGroup/dispatch as possible. How low does it go?

## Prediction

Expansion near 0, encoding about 2 µs per dispatch: under 3 ms for 1,238 dispatches.

## Gate (frozen 2026-11-13, after the baseline, before the measured runs)

First change (the measurement pointed at it): every shader generator memoized (`wgslMemo`): the same arguments return the same WGSL string object, so a dispatch neither rebuilds its source nor rehashes a fresh multi-kilobyte string for the pipeline cache (keyed by source). Arms M (memo) and N (`--no-wgsl-memo`, as before), `bench --contexts 64` (non-resident: expand and encode reported per step) with production flags, order N M M N N M, on the 680M: Kolibri-1 first 8 layers (primary), Qwen3-Coder first 12 layers and Qwen3.5-9B (reported).

1. **Correctness:** tokens identical between arms in every variant (same WGSL text, so bitwise).
2. **Decision (Kolibri-8):** host ms per step = expand + encode of the base variants, median of 3 runs per arm. **Works** if M is at least 50% lower; **Works, partial** if 20–50% (then the frozen dispatch list of the Path is the next step); **Rejected** if under 20%.
3. **Reported:** Qwen3-Coder-12 and Qwen3.5-9B host ms; the A100 step gain is a separate remote run.

## Protocol

Code in `inference_plan.mts`; profile fields already exist (`expandMs`, `encodeMs`).

## Log

- 2026-11-12: Added at user request ("to close the gap with: https://github.com/mlc-ai/web-llm - plan next set of webgpu worker optimization experiments"), in the plan [DO_webgpu_speed.md](DO_webgpu_speed.md).
- 2026-11-13: started at user request ("go ahead with ws-33"). Baseline (680M, `bench --contexts 64`, production flags; `--layers N` added to bench: the first N layers with embedding and head): Qwen3.5-9B expand 1.3 + encode 2.9 ms for 366 dispatches (GPU 87 ms); Kolibri-1 first 8 layers 0.6 + 1.0 ms, 206 dispatches (GPU 15.5); Qwen3-Coder first 12 layers 0.7 + 1.5 ms, 222 dispatches (GPU 17.2). Scaled: about 10 ms of JavaScript per Kolibri token and 9 per Qwen3-Coder token on this CPU; the A100 hosts (slower CPUs) make the MoE models' resident steps host-bound (Kolibri 62.5 ms plain, 35.8 resident; Qwen3-Coder 48.1 and 25.1). Found while measuring: the pipeline cache is keyed by the full WGSL source, and every dispatch regenerated that source from its template.
- 2026-11-13: also fixed on the way (another session's new buffer-layout check, b503b85): `moe_route_softmax` had no layout case (Qwen3-Coder stages failed), repacked formats were unknown, and rows were required to be 4-byte aligned (Qwen3-Coder's Q6_K down experts are 630 bytes per row; the decoders read unaligned words).
- 2026-11-13 22:20–22:27: measured runs under the watchdog, all exit 0 (`runs/experiments/WS-33/omarchy-20261113-222042/drive.sh`).

- 2026-11-13: experiment [146](146-kolibri-host-breakdown-a100.md) (DO-84) measured the A100 host: Kolibri's resident step equals its GPU time (33.1 vs 32.5 ms), so the frozen dispatch list has nothing to hide there.

## Results

| Model (680M) | N host ms | M host ms | M against N | dispatches | GPU ms |
|---|---|---|---|---|---|
| Kolibri-1, layers 0–7 | 1.45 | 1.30 | −10% | 206 | 15.5 |
| Qwen3-Coder, layers 0–11 | 1.85 | 1.40 | −24% | 222 | 17.2 |
| Qwen3.5-9B, all 32 | 3.85 | 2.70 | −30% | 366 | 87.0 |

Gate (by script): 1 passes (tokens identical); 2: −10% on Kolibri-8: **Rejected**.

## What it taught

- Rebuilding WGSL sources per dispatch cost real time where shaders are generated from templates with arguments (Qwen3.5's grouped and coalesced kernels, Qwen3-Coder's), little where they are mostly constant strings (Kolibri's expert and attention kernels).
- The JavaScript side of a step is small on a fast desktop CPU: 1.3 ms for 206 dispatches. If the A100 hosts' MoE steps are host-bound (Kolibri: 62.5 ms plain, 35.8 resident, about 28 ms of GPU work), the cost is either their slower CPUs (several times this) or work outside JavaScript (the GPU process's command decoding and validation), which `encode_ms` does not see. Measuring expand, encode, submit and map on an A100 host (a non-resident bench of Kolibri, with this cache) decides whether a frozen dispatch list (the Path's second step) or fewer dispatches is the lever.

## Limits

- 680M only, layer prefixes of the MoE models (host time per step scales with dispatches); the browser timer quantizes these small times (0.1 ms).
- The A100 host breakdown was not measured here.

## Reproduce

```bash
bash runs/experiments/WS-33/omarchy-20261113-222042/drive.sh
```
