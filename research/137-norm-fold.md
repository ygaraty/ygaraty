# Folding the RMS norm into the grouped product is exact but slower: +6.5% on an A100, +17% on the 680M

| | |
|---|---|
| **Conclusion** | Rejected: norm fold removes 64 launches, tokens identical, but the A100 step grows 17.65 → 18.8 ms (+6.5%, measured) |
| **Status** | executed |
| **ID** | WS-49 (experiment 137) |
| **Category** | browser |
| **Where** | workstation (A100 check bundled with DO-76) |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | The plan executor's peephole: an `rms_norm` (one group, plain) whose output the next grouped product (experiment 135) reads becomes part of that dispatch: each workgroup recomputes the norm's sum of squares with the norm kernel's exact reduction (256 virtual lanes, the same tree) and normalizes the input as it reads it; workgroup 0 still writes the norm's output. Arms: today's default (grouping on) vs grouping + norm fold; Qwen3.5-9B, 680M, then the A100 |
| **Result** | Tokens identical everywhere; dispatches 366 → 302 per token. 680M screen (G F F G G F, resident): 89.35 → 104.65 ms at 64 positions (+17.1%), 90.75 → 105.85 at 1,024. A100 80 GB (G F F G with DO-76): 17.65 → 18.8 ms (+6.5%), 20.35 → 21.55 at 1,024 (+5.9%). Gate 4 (A100 decides): **Rejected**. Every workgroup repeats the norm's 16 KB sum of squares and loads the norm weight beside each input element; that costs more than the 64 launches save. Grouping without the fold stays the default. Shared A100 rental, about $0.55 for three experiments. |
| **Evidence** | measured-local and measured-remote; curated `public/research/evidence/browser/norm-fold-2026-11-13.json`; raw `runs/experiments/WS-49/omarchy-20261113-124118/` and `runs/pods-data/do76/out/`; code 6c73700 (`--norm-fold`, opt-in) |

## Question

On the A100 (experiment 136) the per-layer norms take 2.3 profiled ms per token, the largest family of small kernels, and each sits just before a grouped product that reads its output. Folding them removes 64 launches per token. Does it shorten the step with identical tokens?

## Prediction

Bitwise identical; 680M within noise; A100 −3–6%.

## Gate (frozen 2026-11-13, after one development try, before the measured runs)

Arms G (today's default: `--matvec 2,4,8 --matvec-c 16,4 --q8-fast-up-to 8 --group-matmul`) and F (G plus `--norm-fold`), `bench --contexts 64,1024 --resident`, order G F F G G F, Radeon 680M.

1. **Correctness:** F's tokens identical to G's in every variant.
2. **Launches:** at least 60 fewer dispatches per token (count from a non-resident run per arm).
3. **680M screen:** resident step at 64 positions, median of 3 runs per arm: **passes** if F is not more than 2% longer; otherwise **Rejected on the 680M**.
4. **A100 (bundled with DO-76's rental, if approved):** G F F G resident; **Worth it, now default** if F is at least 3% shorter there, otherwise **Rejected**. The verdict is the A100's if it runs, else the 680M screen's.

## Protocol

Local B F F B B F (resident, 64 and 1,024 positions), then the A100 run with DO-76.

## Log

- 2026-11-13: Added and started at user request ("all levers as experiment and start them"), from experiment [136](136-matvec-group-a100.md)'s A100 profile.
- 2026-11-13: development try (`omarchy-*-try`, non-resident, 64 positions): tokens identical; 366 → 302 dispatches; GPU 87.3 → 103.7 ms (+18%): each workgroup repeats the norm's 16 KB sum of squares, and the inner loop loads the norm weight beside every input element.
- 2026-11-13 12:41–13:01: screen G1 F1 F2 G2 G3 F3 (+ Gc Fc counts), all exit 0: tokens identical in every variant; dispatches 366 → 302 (−64); resident step at 64 positions 89.35 (G) against 104.65 ms (F), **+17.1%** (1,024: +16.6%). 680M screen: **Rejected on the 680M**; the A100 run decides (gate item 4).
- 2026-11-13: User: "Commit, rent, $3 cap, 2h": one 80 GB A100 shared by WS-49, DO-76 and DO-77.
- 2026-11-13 12:09–12:23 UTC: one Vast.ai A100 SXM4 80 GB (Czechia, $1.227/h, driver 595.91.07) at 6c73700, shared by WS-49, DO-76 and DO-77; order G1 F1 R2_1 R4_1 R4_2 R2_2 F2 G2, then count runs Gc Fc; all exit 0. The Kolibri GGUF download failed (HTTP 403: the presigned URL had expired at 23:35 CET the night before), so DO-77's A100 part did not run. Logs copied back; machine deleted by recorded ID at 12:24 (`pods ls`: 0). About $0.55 for the rental, shared.

## Results

| | 680M G | 680M F | A100 G | A100 F |
|---|---|---|---|---|
| resident step, 64 positions (ms) | 89.35 | 104.65 (+17.1%) | 17.65 | 18.8 (+6.5%) |
| resident step, 1,024 positions (ms) | 90.75 | 105.85 (+16.6%) | 20.35 | 21.55 (+5.9%) |
| dispatches per token | 366 | 302 | 366 | 302 |

A100 detail profile (64 positions, profiled ms): the norms fall from 2.07 to 0.54 ms, but the GLU kernel grows 5.10 → 6.36 and the DeltaNet group 2.60 → 3.40; GPU step 17.1 → 18.4 ms.

Gate (by script): 1 passes (tokens identical), 2 passes (−64 dispatches), 3 fails on the 680M (+17.1%), 4 fails on the A100 (+6.5%, needed −3%): **Rejected**.

## What it taught

- A launch is cheaper than redundant work in every workgroup: thousands of workgroups each re-reading the 16 KB row and doing a 256-lane reduction cost more than one norm dispatch, even on the A100.
- The norm is the cost of its reduction, not of its launch: removing 64 launches saved 1.5 ms of norm time but added 2 ms elsewhere. A norm fold would need the sum computed once (in the previous kernel's epilogue, which needs the whole residual row, i.e. a cross-workgroup step).
## Limits

- One A100 host, two runs per arm.
- Only this fold design was tried (sum repeated per workgroup); computing the sum once and passing it on was not.

## Reproduce

```bash
# local: runs/experiments/WS-49/omarchy-20261113-124118/drive.sh
# A100: runs/pods-data/do76/do76.sh (arms G and F)
```

