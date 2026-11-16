# Repacking Q5_K and Q6_K as well cuts the A100 step another 7.8% (73 tokens per second), tokens identical

| | |
|---|---|
| **Conclusion** | Worth it: Q5_K and Q6_K repacked too cut the A100 step 14.8 → 13.65 ms (−7.8%, 73 tok/s), tokens identical (measured) |
| **Status** | executed |
| **ID** | DO-81 (experiment 142) |
| **Category** | browser |
| **Where** | remote (680M screen done) |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | `--repack Q4_K,Q5_K,Q6_K`: besides experiment 141's Q4_KR, Q5_K → Q5_KR (8 scale/min float32 pairs, qh, qs: 224 instead of 176 bytes) and Q6_K → Q6_KR (16 scales d·sc as float32, ql, qh: 256 instead of 210 bytes, every block 4-byte aligned), for layer matrices and the output head; decoders in every kernel table; outputs bit for bit. Arms R4 (Q4_K repacked, experiment 141's winner) and RA (all three), Qwen3.5-9B, bench resident at 64 and 1,024 positions, order R4 RA RA R4, one A100 |
| **Result** | A100 SXM4 40 GB, R4 RA RA R4, resident: step 14.8 → 13.65 ms at 64 positions (**−7.8%**, 73 tokens per second) and 17.3 → 16.15 at 1,024 (−6.6%); tokens identical in every variant; memory 7.74 → 8.30 GB. Detail profile: Q6_K products 3.52 → 3.10 ms, the DeltaNet group (qkv Q5_K) 2.34 → 1.96, ssm_out (Q5_K) 1.43 → 1.08. Gate: **Worth it** (≥ 3%). Together with experiment 141: 17.7 → 13.65 ms (−23%) on this A100 from repacking alone; still opt-in (`--repack Q4_K,Q5_K,Q6_K`) until the per-adapter large-GPU default. About $0.30. |
| **Evidence** | measured-remote and measured-local; curated `public/research/evidence/browser/k-repack-a100-2026-11-13.json`; raw `runs/pods-data/do81/out/` and `runs/experiments/DO-81/omarchy-20261113-1756-screen/`; code 3195523 |

## Question

After experiment 141 the A100 step is 14.75 ms, with Q6_K (3.5 ms: ffn_down in half the layers, attn_v, the output head) the largest product family and Q5_K (attn_qkv, ssm_out) next. Q6_K's 210-byte blocks are not even 4-byte aligned, so every word costs two loads and shifts. Does the same exact repack help them?

## Prediction

Another 5–10% off the A100 step (to about 13.5 ms).

## Gate (frozen 2026-11-13, after the 680M screen, before renting)

1. **Correctness:** RA's tokens identical to R4's in every variant.
2. **Decision (A100):** resident step at 64 positions, median of each arm's two runs: **Worth it** if RA is at least 3% shorter than R4 (joins the large-GPU repack default, follow-up); **Works, small gain** if 1–3%; **Rejected** otherwise (Q5_K/Q6_K stay raw).

## Protocol

One A100 SXM4 40 GB: `bench --contexts 64,1024 --resident --matvec 2,4,8 --matvec-c 16,4 --q8-fast-up-to 8 --group-matmul --repack Q4_K[,Q5_K,Q6_K]`, order R4 RA RA R4, then one non-resident run per arm (detail profile). About 25 minutes, about $0.30; budget asked first.

## Log

- 2026-11-13: Added at user request ("yes" to repacking Q6_K and Q5_K next), from experiment 141. (Drafted as DO-80, renumbered because DO-80 was taken by the Qwen3-Coder plan in the meantime.)
- 2026-11-13: 680M screen (`runs/experiments/DO-81/omarchy-20261113-1756-screen/`): kernels `--cases decode1` with all three repacked: all six matrices' outputs bit-identical to the raw formats (and within 3.3e-7 of NumPy). Full model (bench, 64 and 1,024 positions, grouping on): tokens identical for none, Q4_K and all three repacked; allocated 6.04 / 6.96 / 7.52 GB; GPU at 64 positions 87.1 / 101.4 / 108.9 ms (the 680M is memory-bound: each repack costs there).
- 2026-11-13: User: "yes. do 1 and 2": commit, rent one A100, $1 cap, 60 min.
- 2026-11-13 17:35–17:55 UTC: `do81` (Vast.ai A100 SXM4 40 GB, Mississippi, $0.905/h, driver 595.71.05) at 1567250; R4_1 RA_1 RA_2 R4_2 R4c RAc all exit 0. Logs copied back; machine deleted by recorded ID at 17:55 (an unrelated `ygaraty-panel-paid-…` machine, not made here, left alone).

## Results

| Positions | R4_1 | RA_1 | RA_2 | R4_2 | R4 median | RA median | RA against R4 |
|---|---|---|---|---|---|---|---|
| 64 | 14.9, 14.8 | 13.7, 13.7 | 13.6, 13.4 | 14.7, 14.8 | 14.80 | 13.65 | −7.8% |
| 1,024 | 17.3, 17.3 | 16.4, 16.1 | 16.1, 16.2 | 17.1, 17.3 | 17.30 | 16.15 | −6.6% |

Detail profile, 64 positions (profiled GPU ms), R4 → RA: Q6_K products 3.52 → 3.10; DeltaNet group 2.34 → 1.96; Q5_K ssm_out 1.43 → 1.08; GLU (already Q4_KR) 3.40 → 3.39; norm 2.34 → 2.33. GPU step 14.17 → 12.93 ms.

Gate (by script): 1 passes (tokens identical); 2: −7.8%, at least 3%: **Worth it**.

## What it taught

- Q6_K gains less per byte than Q4_K did (−12% on its products against −33%): its 6-bit values are still assembled from two arrays at every word; only the scales and the misalignment went away. Q5_K gains about 25%.
- The A100's one-conversation step is now 13.65 ms (73 tokens per second) with all three formats repacked, from 17.7 (experiment 141's baseline) and 30 in experiment 123.
- The largest remaining families: the GLU kernel (3.4 ms), Q6_K (3.1), norms (2.3).

## Limits

- One A100 host, two runs per arm; opt-in until the per-adapter default for large GPUs and the contributor's self-test include the repacked formats.
- The 680M is slower with every repack (memory-bound): the default must stay off there.

## Reproduce

```bash
# on an A100 after contributor_fleet setup, at 1567250
bash runs/pods-data/do81/do81.sh
```
