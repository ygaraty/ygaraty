# The exact repack does not speed up Kolibri-1 on an A100 (−0.7%): its step is not bound by the dense products

| | |
|---|---|
| **Conclusion** | Rejected: the repack changes Kolibri-1's A100 step 35.8 → 35.55 ms (−0.7%), tokens identical (measured) |
| **Status** | executed |
| **ID** | DO-82 (experiment 143) |
| **Category** | browser |
| **Where** | remote (local layer screen done) |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | The MoE adapter (`kolibri_gpu.mts`) repacks K-quant matrices and expert chunks at load when `--repack Q4_K,Q5_K,Q6_K` is given (same code as Qwen3.5's, experiments 141/142). Arms B (today's Kolibri flags: `--matvec 2,4,8 --matvec-c 16,4 --q8-fast-up-to 8`) and R (B plus the repack), Kolibri-1 bench resident at 64 and 1,024 positions, order B R R B, one A100 80 GB (shared with DO-80's first A100 run) |
| **Result** | A100 SXM4 80 GB, B R R B, resident: step 35.8 → 35.55 ms at 64 positions (−0.7%) and 110.8 → 110.25 at 1,024 (−0.5%); tokens identical in every variant; memory 46.2 → 60.75 GB. Gate: under 2%: **Rejected**; Kolibri stays raw. Unlike Qwen3.5 (−23%, experiments 141/142), Kolibri's step is not bound by its dense K-quant products: it triples from 64 to 1,024 positions (attention), and its routed experts run through their own per-slot kernel. Shared rental (about $1.90 for three experiments). |
| **Evidence** | measured-remote and measured-local; curated `public/research/evidence/browser/kolibri-repack-a100-2026-11-13.json`; raw `runs/pods-data/do82/out/` and `runs/experiments/DO-77/omarchy-20261113-1304-screen/R`; code e2c7590 |

## Question

On Qwen3.5-9B the exact repack cut the A100 step 23% (experiments 141 and 142): the products are decoding-bound there. Kolibri-1's experts and attention are the same K-quant formats. Does it hold for Kolibri, where the routed experts run through their own kernel?

## Prediction

10–20% shorter at 64 positions; less at 1,024 (attention dominates there, experiment 139).

## Gate (frozen 2026-11-13, after the local screen, before renting)

1. **Correctness:** R's tokens identical to B's in every variant.
2. **Decision:** resident step at 64 positions, median of each arm's two runs: **Worth it** if R is at least 5% shorter (joins the large-GPU repack default); **Works, small gain** if 2–5%; **Rejected** otherwise.

## Protocol

`runs/pods-data/do82/do82.sh` on one A100 80 GB: fetch Kolibri (presigned URL, SHA-256 checked), KB1 KR1 KR2 KB2. Budget: shared $3 cap, 2-hour teardown (with DO-80's run).

## Log

- 2026-11-13: Added and started at user request ("3 also on a100"), from experiments 141 and 142.
- 2026-11-13: local screen (680M, `stage --first 0 --last 3`, `runs/experiments/DO-77/omarchy-20261113-1304-screen/R`): outputs byte-identical to the unrepacked run in all four calls; allocation 2.93 → 3.80 GB for three layers (the full model about 58 GB repacked).
- 2026-11-13 18:40–20:01 UTC: `do82` (Vast.ai A100 SXM4 80 GB, Texas, $1.422/h, driver 610.57.04) at e2c7590, shared by DO-80, DO-82 and DO-83; Kolibri fetched with a matching SHA-256; every step exit 0. Logs copied back; machine deleted by recorded ID at 20:01 (`pods ls`: 0). About $1.90 for the rental, against the $3 cap.

## Results

| Positions | B median | R median | R against B |
|---|---|---|---|
| 64 | 35.80 | 35.55 | −0.7% |
| 1,024 | 110.80 | 110.25 | −0.5% |

Gate (by script): 1 passes (tokens identical); 2: −0.7%, under 2%: **Rejected**.

## What it taught

- The repack pays only where the one-token dense products dominate the step (Qwen3.5-9B). Kolibri's 35.8 ms at 64 positions is mostly elsewhere: its expert kernel (one workgroup row per slot, MR = 4 rows per workgroup) and attention, which triples the step by 1,024 positions.
- Next for Kolibri on large GPUs: the expert kernel and long-context attention, not the weight layout.
## Limits

- One host, two runs per arm; no detail profile was taken for Kolibri here.

## Reproduce

```bash
# on an A100 80 GB after setup and the Kolibri fetch, at e2c7590: the K* steps of runs/pods-data/do82/do82.sh
```

