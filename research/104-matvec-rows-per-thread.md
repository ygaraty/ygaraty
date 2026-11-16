# Rows per thread make 8 decode rows cost 1.3–1.7× one, bit for bit, but no single variant passes: R = 4 slows single tokens

| | |
|---|---|
| **Conclusion** | **Failed gate:** R = 4 brings 8 rows to 1.3–1.7× one row, but slows single tokens up to 1.9× (measured, workstation). |
| **Status** | executed |
| **ID** | WS-15 |
| **Category** | browser |
| **Where** | workstation (Radeon 680M, Chrome WebGPU) |
| **Date** | Added 2026-11-09 |
| **Path** | `webgpu_kernels.mts`: today's matrix-vector kernel (1 row per thread, up to 4 tokens per dispatch row) against `matvecR` (R = 1, 2, 4 rows per thread, VT = 4 or 8 tokens), Qwen3.5-9B's Q4_K `ffn_up`, Q6_K `ffn_down` and Q5_K `attn_qkv` at 1, 2, 4, 5, 8, 9 and 16 tokens |
| **Result** | **Bit-exactness met; go line met by no single variant.** Every variant's 21 outputs are bit-identical to today's kernel. At 8 tokens, R = 4/VT = 8 costs 1.32× / 1.66× / 1.72× today's single token (`ffn_up` / `ffn_down` / `attn_qkv`; today 3.47× / 3.90× / 4.30×), within the 2.0× line, but its single token is 0.93× / 1.87× / 1.56× today's (line: 1.05×). R = 2 makes a single token faster (0.70× / 0.93× / 0.92×) but 8 tokens cost 1.92–2.38× |
| **Evidence** | Measured: [evidence/browser/matvec-rows-per-thread-2026-11-09.json](evidence/browser/matvec-rows-per-thread-2026-11-09.json) (median ms per case and arm, ratios, bit-identity); raw reports and outputs in `runs/experiments/WS-15/omarchy-20261109-2206/` (not in git); gate script `runs/experiments/WS-15/analyze.py`. Kernel `matvecRowsShader` in `webgpu_kernels.mts` (opt-in, `Gpu.matvec`) |

## Question

[DO-25](112-speculative-decoding-rented.md)'s screen found that a speculative checking pass of k + 1 rows costs almost k + 1 decode
steps, and [experiment 93](093-qwen35-interleaved-conversations.md) that batching saves no GPU work: on the
680M the 4-token matrix-vector kernel costs 2.5× the 1-token one even at 2 tokens (ffn_up 0.82 / 1.99 /
2.13 / 3.90 ms at 1 / 2 / 4 / 8 tokens). Each thread reloads every token's inputs for every weight word, and
the 4 rows of a workgroup load the same inputs separately. **Does loading the inputs once for R rows per
thread make 8 rows cost at most 2× one row, with outputs bit-identical to today's kernel?**

## Prediction

Input loads fall R-fold; R = 4 with VT = 8 gives 8 rows in 1.3–2× one row's time, and 1 row slightly faster
than today (fewer redundant loads). Bit-identical outputs, since every row keeps its sums and reduction tree.

## Gate

Frozen 2026-11-09 22:05 CET, before any run of the new kernel.

- **Bit-exactness (required):** every output value of each variant equals today's kernel (`--prefill-kernel
  matvec`, the exact path) bit for bit, on all 21 cases.
- **Go:** some variant's median time (3 runs of `--repeat 20`) at 8 tokens is at most **2.0×** today's kernel
  at 1 token, for each of the three matrices; and at 1 token it is no slower than today's by more than 5%.
- **No-go** otherwise: the kernel is not adopted and DO-25 keeps only adaptive drafting.
- Report also: 16 tokens, the single-token speedup, weight GB/s.

## Protocol

```bash
G=runs/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf
for run in 1 2 3; do
  python -m ygaraty.experiments.qwen35_webgpu kernels --gguf $G --cases decode --prefill-kernel matvec --out OUT/base-$run
  for v in 1,4 2,4 4,4 2,8 4,8; do
    python -m ygaraty.experiments.qwen35_webgpu kernels --gguf $G --cases decode --prefill-kernel matvec --matvec $v --out OUT/r${v/,/x}-$run
  done
done
python runs/experiments/WS-15/analyze.py OUT
```

Runs interleaved by repeat; short submissions (the 680M drives the display).

## Log

- 2026-11-09 22:05 CET: added; gate frozen. Kernel `matvecRowsShader` in `webgpu_kernels.mts` (opt-in through
  `Gpu.matvec`; the default path unchanged), benchmark options `--cases decode` and `--matvec R,VT`.
- 22:06: benchmark started (`runs/experiments/WS-15/omarchy-20261109-2206`).
- 22:11: benchmark finished (18 page runs); gate computed by `analyze.py`: bit-exactness met,
  no variant meets the go line.

## Results

Median ms of 3 runs (`--repeat 20`), and the ratio to today's kernel at 1 token:

| Kernel | Matrix | 1 | 2 | 4 | 8 | 16 tokens |
|---|---|---|---|---|---|---|
| today (R = 1, VT = 4) | ffn_up (Q4_K) | 1.15 | 2.05 (1.79×) | 2.09 (1.83×) | 3.98 (3.47×) | 7.85 (6.86×) |
| | ffn_down (Q6_K) | 1.13 | 2.25 (1.99×) | 2.31 (2.04×) | 4.43 (3.90×) | 8.64 (7.61×) |
| | attn_qkv (Q5_K) | 0.66 | 1.48 (2.23×) | 1.50 (2.28×) | 2.83 (4.30×) | 5.47 (8.29×) |
| R = 2, VT = 4 | ffn_up | **0.80 (0.70×)** | 1.26 | 1.30 | 2.37 (2.07×) | 4.55 |
| | ffn_down | 1.05 (0.93×) | 1.52 | 1.58 | 2.88 (2.54×) | 5.49 |
| | attn_qkv | 0.60 (0.92×) | 0.93 | 0.97 | 1.73 (2.62×) | 3.32 |
| R = 4, VT = 8 | ffn_up | 1.06 (0.93×) | 1.47 | 1.45 | **1.51 (1.32×)** | 2.79 (2.44×) |
| | ffn_down | 2.12 (1.87×) | 1.77 | 1.83 | **1.88 (1.66×)** | 3.48 (3.07×) |
| | attn_qkv | 1.03 (1.56×) | 1.09 | 1.08 | **1.13 (1.72×)** | 2.04 (3.09×) |

R = 1/VT = 4 reproduces today's kernel (0.97–1.02×); R = 2/VT = 8 and R = 4/VT = 4 lie between (see the
evidence file). All 15 variant runs gave outputs bit-identical to today's.

## What it taught

- **The decode product is bound by input loads, not by dequantization.** Loading each input word once for R
  rows cuts 8-token cost from 3.5–4.3× to 1.3–1.7× one token, and R = 2 makes even one token up to 30%
  faster (`ffn_up` 1.15 → 0.80 ms). Experiment 93's reading (dequantization-bound) was wrong.
- **R = 4 starves the GPU on smaller matrices at one token:** `ffn_down` (4,096 rows) and `attn_qkv` get a
  quarter of the workgroups, and their single token slows 1.6–1.9×; with more tokens per workgroup the extra
  arithmetic hides it.
- **The kernel choice should depend on the token count:** R = 2 for one token, R = 4/VT = 8 for 2–8. That
  combination would pass both lines here, but it was chosen after seeing these numbers, so it needs its own
  gate.
- Sums and reduction trees kept per row keep the output bit for bit, as predicted.

## Limits

- One GPU (the 680M); the RTX 3060's balance of loads and arithmetic differs.
- Kernel times only; a stage's step also runs attention, DeltaNet, norms and the head.
- Three matrices of the first layer; the other layers have the same shapes.

## Reproduce

As the Protocol, then `python runs/experiments/WS-15/analyze.py OUT`.
