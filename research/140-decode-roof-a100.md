# On an A100 the one-token products are decoding-bound: the same reads without decoding run 1.71× faster

| | |
|---|---|
| **Conclusion** | Found: decoding-bound; the same access pattern without decoding reads 1.71× faster (198 vs 116 GB/s, measured, A100) |
| **Status** | executed |
| **ID** | DO-78 (experiment 140) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | One A100 (40 GB is enough), headless Chrome, the `kernels` microbenchmark extended with two diagnostic kernels on the same Qwen3.5-9B matrices (Q4_K, Q5_K, Q6_K): (A) the copy probe (727 GB/s, experiment 128); (B) a read-only kernel with exactly matvecC's thread-to-byte mapping and loop, summing the raw words without decoding; (C) today's matvecC. Each timed over more than 40 MB of distinct matrices per timing (rotating buffers, so the A100's L2 does not serve them: the flaw of experiment 134's sweep). Related: [DO-53](DO-53.md) (RTX 3060 roofline) |
| **Result** | A100 SXM4 40 GB, six Qwen3.5-9B matrices, each over more than 64 MiB of copies, two runs: bytes-weighted A (plain read) 370, B (matvecC's access pattern, no decoding) 198, C (production kernel) 116 GB/s. Checksums equal NumPy's for A and B in every case. Gate: B/C = 1.71 (≥ 1.5): **decoding-bound**. Also B/A = 0.54: the access pattern alone loses about half against a plain read, worst on the output head (0.32) and the 4,096-column matrices (0.70–0.84), not on the long ffn rows (1.02–1.04). And a single plain-read dispatch of 19–110 MB reaches only 218–470 GB/s, below the 727 GB/s copy probe. Next: a weight layout that is cheaper to decode (WS-40), then the access pattern for short rows. About $0.17. |
| **Evidence** | measured-remote and measured-local; curated `public/research/evidence/browser/decode-roof-a100-2026-11-13.json`; raw `runs/pods-data/do78/out/`; 680M check `runs/experiments/DO-78/`; code 24e3bb5 (`kernels --roof-mb`) |

## Question

Today's default decodes Qwen3.5-9B at 17.65 ms per token on an A100 (experiment 138), with the products reading at 330–410 GB/s against Dawn's 727 GB/s copy limit. Three guesses at the cause failed (experiments 134, 137, 138). Is the limit the decoding arithmetic (then a simpler weight layout helps, WS-40) or the access pattern and occupancy (then a different kernel structure is needed)?

## Prediction

B near 600–700 GB/s (decoding is the limit) is the more likely outcome; B near C would point at the access pattern.

## Gate (frozen 2026-11-13, after the 680M check, before renting)

As built (`kernels --cases decode1 --matvec-c 16,4 --roof-mb 64 --rounds 5`): per one-token case (ffn_up, ffn_gate, attn_q: Q4_K; attn_qkv: Q5_K; ffn_down, output: Q6_K), enough copies of the matrix for more than 64 MiB, 30 dispatches per timing cycling over the copies, rounds in alternating order A B C / C B A; GB/s = matrix bytes / median ms per dispatch.

1. **Validity:** A's and B's checksums equal NumPy's from the file's bytes for every case (B: the XOR of each row's quant words; words matvecC reads an even number of times cancel).
2. **Diagnostic, Found**, judged on the bytes-weighted means over the six cases: **decoding-bound** if B ≥ 1.5 × C; **access-bound** if B ≤ 1.2 × C; otherwise **mixed**. Also reported: B / A (how far matvecC's access pattern is from a plain read).

## Protocol

Workstation first: implement B and the rotating-buffer timing; check on the 680M that B sums the same words C reads (a checksum against NumPy). Then one A100 rental (about 25 minutes, about $0.30) after budget approval.

## Log

- 2026-11-13: Added at user request ("Yes", after "Next promising result?"), from experiments 134, 137 and 138.
- 2026-11-13: workstation part done: A (16-byte streaming read with XOR), B (matvecC's mapping, 16 threads per row, 4 per item, raw words XORed and inputs summed) and C (production kernel) in `kernels --roof-mb`. 680M check (`runs/experiments/DO-78/omarchy-*-try`): checksums equal for A and B on all six cases; A 62–67, B 63–65, C 56–65 GB/s (near the 62 GB/s copy roof, as expected there).
- 2026-11-13: User: "Commit, rent, $1 cap, 60 min".
- 2026-11-13 14:52–15:01 UTC: `do78` (Vast.ai A100 SXM4 40 GB, Massachusetts, $0.694/h, driver 580.159.03) at 24e3bb5; model fetch 57 s; R1 and R2 30 s each, exit 0. Logs copied back; machine deleted by recorded ID at 15:01 (`pods ls`: 0).

- 2026-11-13: acted on in experiment [141](141-q4k-repack-a100.md) (DO-79): the exact Q4_K repack cuts the A100 step 16.7%.

## Results

Mean of R1 and R2, GB/s (matrix bytes / median ms per dispatch over 5 rounds of 30 dispatches cycling over the copies):

| Matrix | Format | Copies | A: plain read | B: access pattern | C: production kernel | B/C | B/A |
|---|---|---|---|---|---|---|---|
| ffn_up | Q4_K | 3 | 309.8 | 320.6 | 201.4 | 1.59 | 1.04 |
| ffn_gate | Q4_K | 3 | 320.6 | 326.7 | 180.7 | 1.81 | 1.02 |
| ffn_down | Q6_K | 2 | 467.6 | 334.8 | 252.8 | 1.32 | 0.72 |
| attn_qkv | Q5_K | 3 | 266.2 | 224.3 | 144.4 | 1.55 | 0.84 |
| attn_q | Q4_K | 4 | 217.8 | 153.0 | 89.3 | 1.71 | 0.70 |
| output (rows 0–32,767) | Q6_K | 1 | 468.5 | 148.8 | 83.0 | 1.79 | 0.32 |
| **bytes-weighted** | | | **369.6** | **197.7** | **115.8** | **1.71** | **0.54** |

Gate (by script): 1 passes (A and B checksums equal NumPy's in every case and run); 2: B/C = 1.71 ≥ 1.5: **decoding-bound**.

## What it taught

- **Decoding costs the most:** with the same threads reading the same bytes, dropping the dequantization (scales, mins, nibble unpacking, the multiply-adds) makes every matrix 1.3–1.8× faster. A layout that is cheaper to decode (WS-40: plain 4-bit groups with a precomputed scale and offset, or the scales unpacked at load) is the next lever for one-conversation speed.
- **The access pattern costs too, on short rows:** B matches a plain read on the 12,288-column ffn rows but loses 16–30% on 4,096-column matrices and 68% on the output head (32,768 rows of 4,096 columns): 16 threads per row with 4,096 columns leave each thread few iterations, so a workgroup issues few loads before it reduces. A kernel shape that gives short-row matrices more columns per thread (or more rows per workgroup) is the second lever.
- **One dispatch cannot reach the copy roof:** even a plain 16-byte read of a 19–110 MB matrix in one dispatch reads at 218–470 GB/s, against the 727 GB/s copy probe; smaller matrices are further off. Grouping products into fewer, larger dispatches (experiment 136) helped for this reason.
- **The 680M is near its roof for all three** (A 62–67, B 63–65, C 56–65 GB/s): the A100's gaps are invisible on the workstation, so these levers must be judged on the A100.

## Limits

- Isolated dispatches of one matrix at a time; in the model, products also overlap with neighbouring dispatches and the input may stay cached.
- B keeps matvecC's input reads and a float sum but none of the decoding arithmetic or scale reads beyond the block header; it is a lower bound on the access pattern's cost, not an exact copy of it.
- The output head was timed on its first 32,768 rows only (one copy, 110 MB).
- Two runs on one host; some timings are quantized by the browser timer (30 dispatches of 80–1,330 µs).

## Reproduce

```bash
# on an A100 after contributor_fleet setup, at 24e3bb5
bash runs/pods-data/do78/do78.sh
# or directly:
.venv/bin/python -m ygaraty.experiments.qwen35_webgpu kernels --gguf $Q/Qwen3.5-9B-Q4_K_M.gguf --config $Q/config.json \
  --cases decode1 --matvec-c 16,4 --roof-mb 64 --rounds 5 --out OUT
```
