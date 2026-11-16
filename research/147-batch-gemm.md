# Neither vec4 staging, larger register tiles nor repacked weights lift the batch GEMM much: +9% at 32 conversations on an A100

| | |
|---|---|
| **Conclusion** | Rejected: repacked weights lift 32-conversation decode 102.2 → 111.6 tok/s (+9.2%), under the 10% bar (measured, A100) |
| **Status** | executed |
| **ID** | WS-50 (experiment 147) |
| **Category** | browser |
| **Where** | workstation (A100 check after) |
| **Priority** | high |
| **Date** | Added 2026-11-13; run 2026-11-13 and 14 |
| **Path** | The batch path for 8–32 conversations (today: the batched matvec up to 31, experiment 127's GEMM with 64×32 tiles from 32): a GEMM that decodes each weight block once into registers or workgroup memory and reuses it for all tokens of the tile, with larger per-thread register tiles; optionally `shader-f16` arithmetic where offered (wgpu, experiment 128). Arms: today's default vs the new kernel; Qwen3.5-9B, batches 8, 16, 32 with `--batch-check`; 680M first, then an A100 |
| **Result** | 680M screen: vec4 staging bit-identical but no faster; 8×8 register tiles (now compilable) slower, 36.8 against 43.5 tok/s at 32; 128-row tiles exceed workgroup memory. A100 SXM4 40 GB (shared rental with DO-80), GEMM 64×32 with 4×8 per thread, B R R B: tok/s at 8 / 16 / 32 conversations 84.5 / 61.7 / 102.2 raw, 90.6 / 67.6 / 111.6 repacked (+7.2 / +9.5 / **+9.2%**); every batch row's tokens equal its lone conversation's, last logits within 8.4e-5. Gate: under 10% at 32: **Rejected**. Production keeps its batch path (batched matvec below 32 conversations, this GEMM from 32). About $0.18 (half of a $0.35 rental). |
| **Evidence** | measured-remote and measured-local; curated `public/research/evidence/browser/batch-gemm-2026-11-14.json`; raw `runs/pods-data/do85/out/` and `runs/experiments/WS-50/omarchy-20261113-2235-try/`; code 90c45ba (vec4 staging `--gemm …,v` and fewer-threads tiles, both opt-in) |

## Question

Batched decode saturates near 107 tokens per second at 32 conversations on an A100 (experiments 125, 128): about 1.8 TFLOP/s, roughly 10% of its 19.5 TFLOP/s FP32 peak. On the 680M the 32-conversation step is 87% matrix products at about 720 GFLOP/s (experiment 126). Paid contributors serving many users would gain most here. How much of that headroom can a better batch kernel take?

## Prediction

1.5–2.5× at 32 conversations on the A100; less on the 680M.

## Gate (frozen 2026-11-14, before the A100 run)

A100 (40 GB, shared with DO-80's run), Qwen3.5-9B, `bench --batches 8,16,32 --batch-check --batch-kernel gemm --gemm 64,32,4,8` (today's GEMM tile, which production uses from 32 conversations) with production flags: arms B (raw K-quants) and R (B plus `--repack Q4_K,Q5_K,Q6_K`), order B R R B; tokens per second = the bench's conversations × tokens / wall per step.

1. **Correctness:** every batch row's tokens equal to its lone conversation's in every arm (`--batch-check`), last logits within 2e-3 of the largest.
2. **Decision:** tokens per second at 32 conversations, median of each arm's two runs: **Worth it** if R is at least 30% higher than B; **Works, small gain** if 10–30%; **Rejected** if under 10%. 8 and 16 reported.

## Protocol

Workstation: implement, `bench --batches 8,16,32 --batch-check`, then a per-dispatch profile (`--batch-profile`). A100 run after budget approval (can share a rental with DO-78).

## Log

- 2026-11-13: Added at user request ("Yes", after "Next promising result?"), from experiments 125–128.
- 2026-11-13: started at user request ("approve a100 breakdown and ws-50"). First screen on the 680M (`runs/experiments/WS-50/omarchy-20261113-2235-try/`), Qwen3.5-9B, `bench --batches 8,16,32 --batch-check`:
  - GEMM staging as vec4 (`--gemm 64,32,4,8,v`, one load per operand vector instead of four): bit-identical tokens and logits to the scalar staging, **no speed change** (279 / 601 / 736 ms per step at 8 / 16 / 32 both): shared-memory loads are not the limit.
  - At 32 conversations the GEMM (43.5 tok/s) beats the batched matvec the bench uses by default (32.3), as experiment 127 found; production already switches to the GEMM from 32.
  - Per-dispatch profile at 32 (GEMM 64×32, 4×8 per thread): GEMMs 565 ms of a 685 ms GPU step (about 0.9 TFLOP/s, roughly 27% of the 680M's FP32 peak); the DeltaNet recurrence 88 ms (13%), as about 770 small per-conversation dispatches (4,269 dispatches per step in all).
  - Larger register tiles: 8×8 and 128-row tiles produce invalid WGSL (the generator's decode split needs at least as many threads as tile rows × 8 / words); 64×32 with 8×4 per thread is valid and no faster (43.2 tok/s).
  - Repacked weights with the GEMM: the 680M runs out of memory headroom (repacked model plus batch buffers), so the batch-plus-repack combination needs an A100.
  - Next: (a) one batched DeltaNet recurrence dispatch for all conversations of a step (about 770 launches and up to 88 ms at 32); (b) a GEMM with 8×8 register tiles and a decode split that fits them; judged against the frozen 30% bar on an A100.
- 2026-11-13: the GEMM generator now lets each thread decode whole rows when a tile has fewer threads than rows, so 8×8 register tiles compile (existing tiles unchanged). 680M, 32 conversations: 64×32 with 8×8 per thread (32 threads) 36.8 tok/s against 43.5 for 4×8 (64 threads), vec4 or not; 128-row tiles exceed the 16 KB workgroup memory. On the 680M today's tile is the best of this design; tokens and logits equal the lone conversation's in every arm. The weight decode (each staged value used for 32 tokens) is the remaining suspect, as for one token in experiment 140: next, the GEMM with repacked weights (`--repack`) on an A100 (the 680M runs out of memory with repacked weights at 16–32 conversations).
- 2026-11-13 23:16–23:44 UTC: `do85` (Vast.ai A100 SXM4 40 GB, Massachusetts, $0.70/h, driver 580.159.03, driver 615 excluded), shared with DO-80; B1 R1 R2 B2 all exit 0. Logs copied back; machine deleted by recorded ID at 23:44 (`pods ls`: 0).

## Results

| Conversations | B tok/s (B1, B2) | R tok/s (R1, R2) | R against B |
|---|---|---|---|
| 8 | 84.3, 84.7 | 88.4, 92.7 | +7.2% |
| 16 | 61.4, 62.0 | 65.9, 69.2 | +9.5% |
| 32 | 101.7, 102.7 | 107.6, 115.6 | +9.2% |

Gate (by script): 1 passes (32/32 rows equal, logits within 8.4e-5); 2: +9.2% at 32 (needed 10% for a small gain, 30% for Worth it): **Rejected**.

## What it taught

- The batch GEMM is not decode-bound the way one-token products are: unpacking the scales removes most decoding work yet gains only 9% (one token: 23%, experiments 141/142). Each decoded weight serves 32 tokens, so decoding is a small share; the arithmetic and staging dominate (about 27% of FP32 peak on the 680M).
- Forcing the GEMM at 16 conversations is slow (61.7 tok/s, below 8 conversations): half-empty 32-token tiles. Production's batched matvec below 32 avoids this.
- A big gain would need a different arithmetic path: tensor cores (subgroup-matrix, a Chromium experiment, experiment 128) or f16 arithmetic (wgpu's shader-f16, experiment 128), not tuning this kernel.

## Limits

- One A100 host, two runs per arm; the 680M could not test the repacked GEMM at 16–32 conversations (memory).
- The batched DeltaNet recurrence (13% of the 680M's 32-conversation step) was not built: its state traffic (about 3 GB per step at 32) bounds what one dispatch could save.

## Reproduce

```bash
# on an A100 after contributor_fleet setup, at 90c45ba: the B*/R* steps of runs/pods-data/do85/do85.sh
```
