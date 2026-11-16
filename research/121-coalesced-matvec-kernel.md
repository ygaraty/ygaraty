# A coalesced one-token matrix-vector kernel reads K-quants at 81–99% of the 680M's copy roof; Qwen3.5-9B decode 23% faster, reference-exact

| | |
|---|---|
| **Conclusion** | Works: coalesced matvec at 81–99% of the copy roof; Qwen3.5-9B decode 125.7 → 96.5 ms on the 680M (measured) |
| **Status** | executed |
| **ID** | WS-34 (experiment 121) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Request `subgroups` (and `shader-f16` where present) on the device; K-quant blocks (Q4_K, Q5_K, Q6_K, Q8_0) repacked at load into a layout with aligned 16-byte loads per thread; row reductions with `subgroupAdd`; arms: current kernel vs new, per format, 680M; fallback kept for devices without subgroups |
| **Result** | Works (gate by script). Kernel throughput on the 680M, median of 3 runs per arm (C against D): Q4_K 50.7 against 39.6 GB/s (81% of the 62.3 GB/s roof, 1.28×), Q5_K 53.1 against 44.2 (85%, 1.20×), Q6_K 61.8 against 43.4 (99%, 1.42×); errors at most 3.3e-7 of max |y|. Qwen3.5-9B decode (bench, 64 positions): 125.7 → 96.5 ms per token (8.0 → 10.4 tokens per second), GPU 115.8 → 88.5 ms, matrix-vector products at 62–68 GB/s; at 1,024 positions 126.5 → 98.1. Reference check (Qwen3.5-9B, one stage, C kernel; run on the WS-32 A100 because the reference needs about 36 GB of host memory): 32/32 tokens equal to the NumPy reference, logits within 2.5e-5 of the largest; 41 ms per token on that A100. Against experiment 118: 10.4 tokens per second is 85% of llama.cpp's 12.3 and above WebLLM's 8.9. | |
| **Evidence** | measured-local (kernels, bench) and measured-remote (reference check); curated `public/research/evidence/browser/webgpu-coalesced-matvec-680m-2026-11-12.json`; raw `runs/experiments/WS-34/omarchy-20261112-1430/` and `runs/pods-data/ws32/ws34/`; code 9cd1359 (files copied to the A100: webgpu_kernels.mjs 5b03e389…, qwen35_parity.mjs ad1f752a…, qwen35_webgpu.py a7204381…) |

## Question

Matrix-vector products are the largest share of decode GPU time (Q6_K alone 29% on Kolibri). Our kernels reduce through workgroup memory and read packed bytes with u32 loads. How close to the bandwidth roof do subgroup kernels on a repacked layout come?

## Prediction

From about 25–35% to 60–75% of peak on the 680M; a decode step 1.5–2× faster on Qwen3.5-9B.

## Gate (frozen 2026-11-12, after kernel exploration, before the measured runs)

Arms on the 680M: **D**, the sessions' default one-token kernel (`--matvec 2,4,8`: 2 rows per thread), and **C**, the coalesced one-token kernel (`matvecC`, 16 threads per row, 4 threads per 64-element pair; chosen from TPR,L in {16,32,64} × {1,2,4,8} during exploration, see Log).

1. **Kernel throughput:** `qwen35_webgpu kernels --cases decode1 --repeat 50` on Qwen3.5-9B (Q4_K ffn_up, ffn_gate, attn_q; Q5_K attn_qkv; Q6_K ffn_down and 32,768 rows of the head), D C C D C D (3 runs per arm); per format the median GB/s, as a share of experiment 119's copy roof (62.3 GB/s).
2. **Decode:** `bench --contexts 64,1024` for Qwen3.5-9B, D then C; step ms and matrix-vector GB/s per format.
3. **Correctness:** every kernel case within 1e-5 of the float64 product (relative to max |y|); `pipeline --splits 0,32 --generate 32` on Qwen3.5-9B with C: 32/32 tokens equal to the NumPy reference, logits within 2e-3 of the largest.

**Works** if C reaches at least 60% of the roof, or 1.5× D's GB/s, for each of Q4_K, Q5_K and Q6_K, with correctness. Otherwise **Rejected** (a correctness break: **Failed gate**). The decode step change is reported, not gated.

## Protocol

Kernels in `webgpu_kernels.mts` (feature-detected), repack in the stage loaders, self-test cases extended; WS-31 bench before and after.

## Log

- 2026-11-12: Added at user request ("to close the gap with: https://github.com/mlc-ai/web-llm - plan next set of webgpu worker optimization experiments"), in the plan [DO_webgpu_speed.md](DO_webgpu_speed.md).
- 2026-11-12: exploration on the 680M (`kernels --cases decode1 --repeat 50`, not the gate's runs): D 38–44 GB/s; coalesced variants TPR,L: 16,1 ≈ D; 32,2 40–53; **32,4 and 16,4 best (Q4_K 48–51, Q5_K 53–54, Q6_K 59–64 GB/s)**; 64,2 and 64,8 and 32,8 and 16,8 lower; scales loaded as three words (an array indexed at run time) slower, reverted. Errors about 2e-7 throughout.
- 2026-11-12 14:30: gate part 1 (`runs/experiments/WS-34/kernels.sh`, D C C D C D): all exit 0.
- 14:38–14:40: part 2 (`decode.sh`): bench D then C, both exit 0. Part 3 on the workstation failed twice: first the scratch environment lacked `transformers`; then the reference (Qwen3.5-9B in fp32, about 36 GB) ran the 28 GB workstation out of memory and was stopped. With the user's approval it ran on the WS-32 A100 (503 GB of host memory) after copying the three changed files (hashes above) and installing CPU torch and transformers: exit 0, 32/32 tokens.

- 2026-11-13: **now the default** for Qwen3.5 and Kolibri-1 sessions (`kernel_defaults` in `qwen_protocol.py`, used by `prepared.py` and `qwen35_session.py`; `YGARATY_KERNELS=off` switches back). Local check, Qwen3.5-9B, 4 batched conversations: self-test passed (worst kernel 5.0e-7 both ways), all 4 conversations' tokens identical to the old kernels, steady 24.5 against 15.5 tokens per second (`runs/experiments/defaults-check/omarchy-20261113-014740/`).

## Results

| Format | D, sessions' kernel (GB/s) | C, coalesced (GB/s) | C / roof | C / D |
|---|---|---|---|---|
| Q4_K | 39.6 | 50.7 | 81% | 1.28 |
| Q5_K | 44.2 | 53.1 | 85% | 1.20 |
| Q6_K | 43.4 | 61.8 | 99% | 1.42 |

Qwen3.5-9B on the 680M (bench): 64 positions 125.7 → 96.5 ms per token; 1,024 positions 126.5 → 98.1 ms.

## What it taught

- The old kernels gave each thread its own 32-byte unit, so neighbouring threads read 32 bytes apart; four threads per unit, each reading consecutive words, lets the memory system serve whole lines. That alone took the matrix-vector products to the roof on this GPU.
- Loading the 12 scale bytes as three words through a run-time-indexed array was slower (the array spills); per-byte loads stay.
- On the 680M our decode is now at 85% of native llama.cpp; the rest is the non-matvec work (6 ms) and host time.

## Limits

- Opt-in (`gpu.matvecC`, `--matvec-c 16,4`); not yet the default in sessions: making it default needs the contributors' admission self-test to cover it and a Kolibri and dense reference check (only Qwen3.5-9B was checked). Not bit for bit with the other kernels (summation order), so DO-25's exact-checking pass must keep using one kernel per conversation.
- One GPU; the first kernel case in each run (ffn_up) is noisy in both arms (warm-up), so the per-format medians pool the cases.
- The kernel only serves one token; batched decode and checking passes keep the R-rows kernel.

## Reproduce

```bash
bash runs/experiments/WS-34/kernels.sh OUT   # kernels --cases decode1 --repeat 50, D C C D C D
bash runs/experiments/WS-34/decode.sh OUT    # bench D, C; pipeline C against the reference (needs ~40 GB host memory)
```
