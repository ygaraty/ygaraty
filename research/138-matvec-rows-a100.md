# Several rows per thread make the one-token products slower on an A100 (+17% with 2, +65% with 4), tokens identical

| | |
|---|---|
| **Conclusion** | Rejected: 2 rows per thread lengthen the A100 step 17.65 → 20.65 ms (+17%), 4 rows +65%; tokens identical (measured) |
| **Status** | executed |
| **ID** | DO-76 (experiment 138) |
| **Category** | browser |
| **Where** | remote (680M screen first) |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | The grouped kernel (experiment 135) with R rows per thread (`--matvec-c 16,4,64,R`): each row slot computes R consecutive rows from the same input loads, each row's sum in matvecC's order (bit for bit); single products also go through it when R > 1 (residual add and row offsets kept). Chosen over wider loads from experiment 136's profile: the GLU kernel (two rows per thread) read gate and up at about 330 GB/s against 257 for one row. Arms R1 (today's default), R2, R4; Qwen3.5-9B |
| **Result** | A100 80 GB, resident step at 64 positions (median of two runs): R1 (today) 17.65, R2 20.65 (+17.0%), R4 29.2 ms (+65.4%); 1,024 positions 20.35, 23.15, 31.8. Tokens identical in every variant (bit for bit, as designed). 680M: GPU 87.3, 95.6, 117.3 ms. Gate: **Rejected**; one row per thread stays. Today's default reaches 17.65 ms (57 tokens per second) on this A100. Shared A100 rental, about $0.55 for three experiments. |
| **Evidence** | measured-remote and measured-local; curated `public/research/evidence/browser/matvec-rows-a100-2026-11-13.json`; raw `runs/pods-data/do76/out/` and `runs/experiments/DO-76/omarchy-20261113-1301-try/`; code 6c73700 (`--matvec-c 16,4,64,R`, opt-in) |

## Question

Experiment 134 showed the coalesced kernel's 57 shapes cannot get near the A100's 727 GB/s copy limit (Q4_K about 257–330 GB/s, Q6_K about 410 in the model); experiment 136 left Q6_K (3.6 ms) and the remaining Q4_K/Q5_K products as the largest costs. Is the kernel's structure (4-byte loads, workgroup-memory reduction) the limit?

## Prediction

Q4_K above 450 GB/s on the A100; step −10–20%; neutral on the 680M.

## Gate (frozen 2026-11-13, before renting)

1. **Correctness:** R2 and R4 tokens identical to R1's in every variant (bit-for-bit by construction).
2. **Decision (A100):** resident step at 64 positions, median of each arm's two runs (order R1 F R2 R4 R4 R2 F R1, shared with WS-49's arms G = R1 and F). **Worth it** if the best of R2/R4 is at least 5% shorter than R1: it becomes the setting for large GPUs (a per-adapter default, implemented as a follow-up; the 680M keeps R1); **Works, small gain** if 2–5%; **Rejected** if under 2%.
3. **680M (reported):** R2 and R4 GPU ms against R1.

## Protocol

680M: kernels microbenchmark for correctness and speed, bench; then one A100 (80 GB, shared with DO-77) after budget approval.

## Log

- 2026-11-13: Added and started at user request ("all levers as experiment and start them"), from experiment [136](136-matvec-group-a100.md)'s A100 profile.
- 2026-11-13: implemented as rows per thread instead of wider loads (see Path). 680M try (non-resident, 64 positions): tokens identical for R1, R2, R4; GPU 87.3 (R1), 95.6 (R2, +9.5%), 117.3 ms (R4, +34%): more rows per thread are slower on the 680M.
- 2026-11-13: User: "Commit, rent, $3 cap, 2h": one 80 GB A100 shared by WS-49, DO-76 and DO-77.
- 2026-11-13 12:09–12:23 UTC: one Vast.ai A100 SXM4 80 GB (Czechia, $1.227/h, driver 595.91.07) at 6c73700, shared by WS-49, DO-76 and DO-77; order G1 F1 R2_1 R4_1 R4_2 R2_2 F2 G2, then count runs Gc Fc; all exit 0. The Kolibri GGUF download failed (HTTP 403: the presigned URL had expired at 23:35 CET the night before), so DO-77's A100 part did not run. Logs copied back; machine deleted by recorded ID at 12:24 (`pods ls`: 0). About $0.55 for the rental, shared.

## Results

| Positions | R1 | R2 | R4 |
|---|---|---|---|
| 64 | 17.65 | 20.65 (+17.0%) | 29.2 (+65.4%) |
| 1,024 | 20.35 | 23.15 (+13.8%) | 31.8 (+56.3%) |

Gate (by script): 1 passes (tokens identical for R2 and R4), 2: the best (R2) is 17% longer: **Rejected**.

## What it taught

- Two rows of different matrices per thread (the GLU kernel) help, two consecutive rows of the same matrix do not: in the GLU kernel the row slots of a workgroup still read neighbouring rows of each matrix, while R consecutive rows per slot spread a workgroup's simultaneous loads over R times as many rows (strided by the row size), and the R copies of the decoder state cost registers. The gain in experiment 136 came from fewer launches and the shared input, not from rows per thread as such.
- With today's defaults Qwen3.5-9B decodes at 17.65 ms per token on this A100 (57 tokens per second).
## Limits

- One A100 host, two runs per arm.
- Rows were consecutive; interleaved rows (slot r takes rows r, r + S, ...) and wider loads per thread (the original idea) were not tried.

## Reproduce

```bash
# A100: runs/pods-data/do76/do76.sh (arms G = R1, R2_*, R4_*)
```

