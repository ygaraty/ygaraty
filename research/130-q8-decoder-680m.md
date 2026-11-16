# A vectorized Q8_0 decoder makes Qwen3.5's small Q8_0 products 2.3× faster on the 680M, short of the 3× gate

| | |
|---|---|
| **Conclusion** | Rejected: Q8_0 products 1.62 → 0.70 ms per token, 2.3× (needed 3×), exact (measured, 680M); opt-in |
| **Status** | executed |
| **ID** | WS-46 (experiment 130) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-13 |
| **Path** | A Q8_0 decoder for the vectorized and coalesced matrix-vector kernels (`PAIR`/`UNIT` entries, blocks of 32 int8 with one f16 scale), so Q8_0 matrices leave the generic `matmul_Q8_0` kernel; Qwen3.5-9B (its Q8_0 tensors); arms: current vs new |
| **Result** | Qwen3.5-9B's Q8_0 matrices (`ssm_alpha`, `ssm_beta`: 48 per token, 32 × 4096) through a Q8_0 entry of the vectorized decoders instead of the generic kernel: GPU ms per token 1.62 → 0.70 in detail profiles (2.3×; the gate needed 3×), a step 0.8–1% shorter. Correctness passes: every kernel case within 3.0e-7 of float64 (1, 2, 3, 8, 9, 32, 64 tokens), tokens identical. At 9 or more tokens (the tiled kernel) the new path is slower (0.24–0.28 against 0.18–0.20 ms). **Rejected** as frozen; the decoder stays in the code as an opt-in (`gpu.fastQ8`, `--q8-fast`). Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/q8-decoder-680m-2026-11-13.json`; raw `runs/experiments/WS-46/omarchy-20261113-020117/` |

## Question

Experiment 123's profile on the A100: `matmul_Q8_0` takes 2.98 ms per token for 20.8 MB (7 GB/s), about 8% of a 34 ms step, because Q8_0 has no fast decoder and falls back to the generic kernel. Does a vectorized decoder bring it near the other formats' rate?

## Prediction

Q8_0 at the K-quant kernels' share of the copy roof; about 2.5 ms per token saved on the A100, less on the 680M (1.6 ms there in experiment 126's profile).

## Gate (frozen 2026-11-13, before the first run)

Workstation (680M), Qwen3.5-9B; arms G (`--q8-generic`: Q8_0 through the generic kernel, as before) and F (the new Q8_0 decoder), both with `--matvec 2,4,8 --matvec-c 16,4 --matvec-b 8`.
1. **Correctness:** `kernels --cases q8` (ssm_alpha and ssm_beta at 1, 2, 3, 8, 9, 32, 64 tokens: coalesced, batched, tiled and the GEMM's tiled fallback): every F case within 1e-5 of the float64 reference relative to the largest output (the admission self-test's tolerance); `bench --contexts 64` decoded tokens identical between arms. (The full NumPy reference check needs about 40 GB of host memory, more than this workstation has; experiment 121's reference check covered the K-quant path.)
2. **Speed:** in `bench --contexts 64` detail profiles (order G F F G), the Q8_0 matrices' GPU ms per token (dispatches named `*_Q8_0`) at least 3× lower in F than in G (medians of the two runs per arm).
Both: **Works**, and the decoder stays on (it is used whenever the kinds allow); otherwise **Rejected** with the numbers.

## Protocol

Kernel cases first (`qwen35_webgpu kernels`), then `bench --contexts 64` detail profiles, then the reference check.

## Log

- 2026-11-13: Added at user request ("anything to gain more raw performance?", answer "Add: Q8_0 fast path").

- 2026-11-13: started at user request ("also WS-46 ... free, on the workstation"). The Q8_0 matrices are only `ssm_alpha` and `ssm_beta` (24 each, 32 × 4096, about 139 KB each): the cost is 48 small dispatches through the generic kernel, not decoding bandwidth. Code: a Q8_0 entry in `PAIR` (a 256-element unit = 8 Q8_0 blocks; pair g = blocks 2g and 2g+1), `fastKind` accepts Q8_0, the GEMM (no Q8_0 `UNIT` decoder) falls back to the tiled kernel for Q8_0, `gpu.fastQ8` / `--q8-generic` switches back; `--cases q8`.

- 2026-11-13 02:0x: `runs/experiments/WS-46/run.sh` into `runs/experiments/WS-46/omarchy-20261113-020117/` (watch.py `finished`): kernel cases F and G, then bench G F F G. The switch was then `--q8-generic` (the new decoder on by default); after the gate failed it became opt-in (`--q8-fast`, `gpu.fastQ8 = false` by default).

## Results

| | G (generic, as before) | F (Q8_0 decoder) |
|---|---|---|
| Q8_0 GPU ms per token (detail-64, two runs each) | 1.617, 1.632 | 0.698, 0.704 |
| GPU ms per token (detail-64) | 88.6, 88.9 | 87.8, 87.3 |
| kernel case ms, 1 / 8 / 32 tokens (ssm_alpha) | 0.076 / 0.222 / 0.190 | 0.056 / 0.050 / 0.266 |
| largest kernel error / largest output | – | 3.0e-7 |

Tokens of base-64 identical in all four runs. Gate 1 passes; gate 2: 2.32× (needed 3×). **Rejected.**

## What it taught

- The generic kernel's cost was not only decoding: 48 dispatches of 32 rows each. The vectorized decoder gives each matrix only 8 workgroups (4 rows each), so what remains is mostly launch time and little parallelism, not bytes.
- Batched decode steps (2–8 tokens) gain most (0.22 → 0.05 ms at 8), prefill chunks (9+ tokens, tiled) lose.
- Next, if it matters: merge `ssm_alpha` and `ssm_beta` into one 64-row product per layer (24 fewer dispatches per token), or split the 4,096 columns over more workgroups. On the A100 the generic kernel took 2.98 ms (experiment 123): a larger share there, so worth measuring on a rental.

- **Follow-up, experiment [132](132-q8-decode-only-680m.md) (WS-47):** used only for calls of at most 8 tokens, the decoder is never slower and is now the default.

## Limits

- 680M only; the A100, where this was 8% of a step, not measured.
- No NumPy reference check of the whole model (it needs about 40 GB of host memory); the kernel cases and identical tokens stand in.

## Reproduce

```bash
bash runs/experiments/WS-46/run.sh runs/experiments/WS-46/<host-ts>   # arm G = default, arm F = --q8-fast
```
