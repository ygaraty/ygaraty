# The prefill GEMM decodes 32 conversations 1.34× faster than the batched matvec, but loses at 8 and 16

| | |
|---|---|
| **Conclusion** | Rejected: GEMM 1.34× the batched matvec at 32 conversations (needed 1.5×), half its speed at 8 (measured, 680M) |
| **Status** | executed |
| **ID** | WS-45 (experiment 127) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Qwen3.5-9B on the Radeon 680M, `bench --batches 1,8,16,32 --batch-check`; arms: B = experiment 125's batched matvec (`--matvec-b 8`), G = the prefill GEMM for batch steps (`--batch-kernel gemm --gemm 64,32,4,8`), T = the tiled prefill kernel (`--batch-kernel tiled`); all with `--matvec 2,4,8 --matvec-c 16,4` |
| **Result** | Aggregate tokens per second at 1 / 8 / 16 / 32 conversations, medians of two runs: B (batched matvec) 10.3 / 30.9 / 31.9 / 32.1; G (prefill GEMM, 64×32 tiles) 10.3 / 14.9 / 26.4 / 43.2; T (tiled) 10.3 / 12.2 / 22.3 / 22.7. Correctness passes (every row the lone conversation's tokens; logits within 7.4e-5 for G, 1.7e-4 for T). Speed fails as frozen: G at 32 is 1.34× B (needed 1.5×) and G at 8 is half of B. A choice by batch size (B up to 16, G at 32) would give 43.2 at 32. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/batch-kernels-680m-2026-11-13.json`; raw `runs/experiments/WS-45/omarchy-20261113-001117/`; code 6801b82 |

## Question

Experiment 126: at 8–32 conversations the batched matvec is arithmetic-bound at about 720 GFLOP/s on the 680M (87% of the step), while the prefill GEMM reaches 1.4–1.9 TFLOP/s at 64 tokens there (its tuning note). The batch path forces the matrix-vector kernels to keep a batch row equal to a lone conversation. Does a many-token kernel serve 8–32 conversations faster, within tolerance?

## Prediction

G about 2× B's matrix-product time at 32 (step about 1.6× faster), less at 8; T between.

## Gate (frozen 2026-11-12, before the first run)

Order B G T T G B, medians of the two runs per arm.
1. **Correctness (G, T):** every batch row's 16 tokens equal the lone conversation's at 8, 16, 32 (the lone run uses the one-token kernel), last logits within 2e-3 of the largest.
2. **Speed:** the better of G and T at 32 conversations gives at least 1.5× B's aggregate tokens per second, and is not slower than B at 8.
Both: **Works**, then an A100 run is proposed (budget asked). Otherwise **Rejected** with the numbers.

## Protocol

Batch sizes 1 (one-token kernels, all arms alike), 8, 16, 32. Batch steps of up to `--matvec`'s third value keep the matrix-vector kernel: 4 for G and T, so 8, 16 and 32 use their kernel.

## Log

- 2026-11-12: Added at user request ("go ahead with next steps"), from experiment [126](126-batch-step-profile-680m.md). Harness: `gpu.batchKernel` and `--batch-kernel` (runBatch uses it instead of forcing "matvec"), `--gemm` passed to bench.

- 2026-11-13 00:0x: rehearsal G at 8 and 32 (`runs/experiments/WS-45/omarchy-20261113-001117/rehearsal-G`): rows equal 8/8 and 32/32, logits within 7.4e-5; 32 conversations 43.2 tokens per second (experiment 126's B: 31.4). It showed a slip in the arms: with `--matvec 2,4,8` batches of up to 8 tokens keep the matrix-vector kernel (not 1–4 as the Protocol said), so G at 8 was the R-rows path (27.8). **Before the screen**, G and T get `--matvec 2,4,4` (batches above 4 use their kernel); B keeps `2,4,8` (its batch path is matvecB either way). Gate unchanged.

- 2026-11-13 00:26: screen finished (watch.py `finished`, 14 min), `runs/experiments/WS-45/screen.sh`.

- 2026-11-13: **now the default** for Qwen3.5 and Kolibri-1 sessions (`kernel_defaults` in `qwen_protocol.py`, used by `prepared.py` and `qwen35_session.py`; `YGARATY_KERNELS=off` switches back). Local check, Qwen3.5-9B, 4 batched conversations: self-test passed (worst kernel 5.0e-7 both ways), all 4 conversations' tokens identical to the old kernels, steady 24.5 against 15.5 tokens per second (`runs/experiments/defaults-check/omarchy-20261113-014740/`).

## Results

| Conversations | B: batched matvec | G: GEMM 64×32 | T: tiled |
|---|---|---|---|
| 1 | 10.3 | 10.3 | 10.3 |
| 8 | **30.9** | 14.9 | 12.2 |
| 16 | **31.9** | 26.4 | 22.3 |
| 32 | 32.1 | **43.2** | 22.7 |

Correctness: G and T rows 8/8, 16/16, 32/32 with the lone conversation's 16 tokens in both runs; last logits within 7.4e-5 (G) and 1.7e-4 (T) of 22.8. Gate 2: G at 32 = 1.34× B (needed 1.5×), G at 8 = 0.48× B. **Rejected.**

## What it taught

- Sharing the inputs across 64 rows (GEMM) beats the batched matvec once the batch fills its 32-token tile: 43.2 against 32.1 at 32 conversations, a step of 741 instead of 997 ms.
- Below a full tile the GEMM wastes its token lanes (a 32-wide tile at 8 tokens does 4× the work): a tile sized to the batch, or a choice by batch size, is needed; the batched matvec stays best up to 16.
- On the 680M even the GEMM stays far below the device's arithmetic peak at 32 tokens (its tuning note: 1.4–1.9 TFLOP/s at 64). The A100, with tensor cores reachable through `subgroup-matrix`, is where a batch kernel can change the curve; this screen does not answer for it.

## Limits

- 680M only; the A100 (experiment 125's machine) not measured with G.
- One GEMM tile (64×32, 4×8 per thread); 16-token tiles not tried.
- Identical prompt in every slot.

## Reproduce

```bash
bash runs/experiments/WS-45/screen.sh runs/experiments/WS-45/<host-ts>   # code 6801b82 or later
```
