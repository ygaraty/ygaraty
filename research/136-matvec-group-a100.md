# Grouped one-token products shorten Qwen3.5-9B's A100 decode step 11.5%, tokens identical: now the default

| | |
|---|---|
| **Conclusion** | Worth it, now default: grouped products cut the A100 step 20.8 → 18.4 ms (−11.5%), tokens identical (measured) |
| **Status** | executed |
| **ID** | DO-75 (experiment 136) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | One rented A100 (SXM4 preferred), headless Chrome, Qwen3.5-9B Q4_K_M, today's production kernel flags (`--matvec 2,4,8 --matvec-c 16,4 --q8-fast-up-to 8`) with (G) and without (B) experiment 135's `--group-matmul`; resident benches B G G B at 64 and 1,024 positions, then one non-resident bench per arm (detail profile, dispatch count) |
| **Result** | A100 SXM4 40 GB, B G G B, resident loop: step 20.8 → 18.4 ms at 64 positions (−11.5%, 54 tokens per second) and 23.5 → 21.0 at 1,024 (−10.6%); tokens identical in every variant. Non-resident: dispatches 518 → 366, GPU 20.4 → 17.9 ms; ffn gate + up + SwiGLU take 5.45 ms in one dispatch where the Q4_K products and silu_mul took more. Gate: **Worth it, now default** for Qwen3.5 (`kernel_defaults`, contributor self-test and fixture); `YGARATY_GROUP_MATMUL=off` (or `YGARATY_KERNELS=off`) switches back; Kolibri unchanged. Production contributor checked on the 680M: 64 tokens identical with and without, 90 ms per token both. About $0.20. |
| **Evidence** | measured-remote; curated `public/research/evidence/browser/matvec-group-a100-2026-11-13.json`; raw `runs/pods-data/do75/out/`; production check `runs/experiments/DO-75/omarchy-20261113-1226-contributor/`; code 5e8584c (experiment 135), default in this commit |

## Question

Experiment 135 put products that read the same input in one dispatch (518 → 366 per token), with identical tokens but no change on the GPU-bound 680M. On an A100, where experiment 134 found short dispatches that cannot fill the GPU, does it shorten the step?

## Prediction

1–3 ms off a 20 ms resident step (5–15%).

## Gate (frozen 2026-11-13, before renting)

1. **Correctness:** G's generated tokens identical to B's in every variant at both lengths.
2. **Decision:** step = the resident variants' wall ms per token at 64 positions, median over each arm's two runs. **Worth it, now default** (in `kernel_defaults` and the contributor, with a switch back) if G is at least 5% shorter; **Works, small gain** (stays opt-in) if 2–5% shorter; **Rejected** if under 2%. Failing 1: **Failed gate**.
3. **Found (reported):** dispatches per token and the detail profile of each arm (what remains outside the large products).

## Protocol

`contributor_fleet up do75 --gpus 1 --gpu "A100 SXM4" --max-hourly 1.6 --minutes 60 --min-ram 64 --disk 100 --location US,CA`, setup, then `runs/pods-data/do75/do75.sh` (tokenizers, model fetch, B1 G1 G2 B2 resident, Bc Gc non-resident). Code 5e8584c. About 35 minutes at about $0.6/h.

## Log

- 2026-11-13: Added and approved at user request ("yes" to an A100 run of experiment 135, $2 cap, 60-minute teardown).
- 2026-11-13: rented `do75` (Vast.ai A100 SXM4 40 GB), setup exit 0 at 5e8584c; `do75.sh` launched.
- 2026-11-13 11:12–11:24 UTC: driver steps all exit 0 (model fetch 32 s, each resident bench about 2 min). Logs copied back; machine deleted by recorded ID at 11:25 (an unrelated `ygaraty-panel-paid-…` machine, not made here, left alone). Cost about $0.20 (17 minutes at $0.694/h).
- 2026-11-13: default for Qwen3.5 in `kernel_defaults` (`group_matmul`), applied by the contributor from the self-test and fixture; production check on the workstation (`qwen35_session local`, one stage, 64 tokens): grouping off and on give identical tokens, 90 ms per token both (the 680M is GPU-bound).

## Results

Resident step, ms per token (each run's two resident variants):

| Positions | B1 | G1 | G2 | B2 | B median | G median | G against B |
|---|---|---|---|---|---|---|---|
| 64 | 20.9, 20.8 | 18.4, 18.4 | 18.2, 18.4 | 20.7, 20.8 | 20.8 | 18.4 | −11.5% |
| 1,024 | 23.6, 23.5 | 21.0, 21.1 | 21.0, 20.8 | 23.5, 23.4 | 23.5 | 21.0 | −10.6% |

Non-resident (Bc, Gc), 64 positions: wall 34.4 → 31.7 ms, GPU 20.4 → 17.9 ms, dispatches 518 → 366. Detail profile (profiled GPU ms): B: Q4_K 11.12, Q6_K 3.82, Q5_K 3.69, norm 2.34, Q8_0 1.57, delta 0.83, attention 0.66, silu_mul 0.46, argmax 0.45. G: GLU (gate, up, SwiGLU) 5.45, Q6_K 3.64, DeltaNet group (qkv, gate, alpha, beta) 2.88, remaining Q4_K 2.46, norm 2.32, remaining Q5_K 1.51, delta 0.82, attention 0.67, argmax 0.45.

Gate (computed by script): 1 passes (tokens identical in every variant, both lengths, resident and not); 2: −11.5%, at least 5%: **Worth it, now default**.

## What it taught

- On the A100 the launch count matters where the 680M did not care: the same change that was −0.3% there is −11.5% here. The GLU kernel reads gate and up together at about 330 GB/s, against 257 for the separate Q4_K products (experiment 134's profile); the alpha/beta products (Q8_0, 32 rows each) disappear into the DeltaNet group.
- What remains outside the large products on the A100: norms 2.3 ms (one per layer, before each group), DeltaNet 0.8, attention 0.7, argmax 0.45. Fusing each norm into the following group (the group reads the normalized input) is the next launch-count lever; Q6_K (3.6 ms at about 410 GB/s) is the largest single product family.
- With today's defaults Qwen3.5-9B decodes one conversation at 54 tokens per second on this A100 (18.4 ms), from 30 in experiment 123.

## Limits

- One host, two runs per arm; Chrome only (Node + Dawn should behave the same, as in experiment 128).
- Kolibri (same executor) not measured: the default applies to Qwen3.5 only.
- The production check ran on the 680M; the A100 gain was measured with the bench, not through the contributor.

## Reproduce

```bash
# on an A100 after contributor_fleet setup, at 5e8584c
bash runs/pods-data/do75/do75.sh
```
