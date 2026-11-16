# Unpacking Q4_K's scales at load cuts the A100 decode step 16.7% despite a third more bytes, tokens identical

| | |
|---|---|
| **Conclusion** | Worth it: exact Q4_K repack cuts the A100 step 17.7 → 14.75 ms (−16.7%, 68 tok/s), tokens identical (measured) |
| **Status** | executed |
| **ID** | DO-79 (experiment 141) |
| **Category** | browser |
| **Where** | remote (680M screen done) |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | Layer Q4_K matrices repacked at load into Q4_KR (`--repack-q4k`): per 256-element block the 8 sub-block scales d·sc and mins dm·m as float32 (exact: an f16 times a 6-bit integer fits in float32), then the 128 quant bytes; 192 instead of 144 bytes. Decoders for every kernel table (one-token, batched, grouped, GEMM, generic) read the floats instead of unpacking 6-bit scales; every product is the same, so outputs are bit for bit. Arms G (today's default, grouping on) and R (G plus the repack), Qwen3.5-9B, bench resident at 64 and 1,024 positions, order G R R G, one A100. Related: WS-40 (a lossy 4-bit layout, same goal) |
| **Result** | A100 SXM4 40 GB, G R R G, resident: step 17.7 → 14.75 ms at 64 positions (**−16.7%**, 68 tokens per second) and 20.45 → 17.35 at 1,024 (−15.2%); tokens identical in every variant; memory 6.83 → 7.74 GB. Non-resident GPU 17.2 → 14.2 ms: the GLU kernel 5.14 → 3.40 ms, the remaining Q4_K products 2.28 → 1.52, the DeltaNet group 2.70 → 2.34, although each now reads 33% more bytes. On the 680M (memory-bound) the same change is +16%. Gate: **Worth it**; as frozen, it becomes a per-adapter default for large GPUs in a follow-up (the 680M keeps raw Q4_K); today it is opt-in (`--repack-q4k`). About $0.30. |
| **Evidence** | measured-remote and measured-local; curated `public/research/evidence/browser/q4k-repack-a100-2026-11-13.json`; raw `runs/pods-data/do79/out/` and `runs/experiments/DO-79/omarchy-20261113-1638-try/`; code 3d45f79 |

## Question

Experiment 140 found the one-token products decoding-bound on an A100: the same access pattern without decoding reads 1.71× faster. Unpacking Q4_K's 6-bit scales and mins at load removes the costliest decode steps (byte loads, shifts, the branchy scale_min, f16 conversions) at the price of 33% more Q4_K bytes. Which wins on the A100?

## Prediction

Uncertain: decoding cost removed is maybe 20–40% of Q4_K's time; extra bytes cost up to 33% where memory-bound. Expect −5% to +10% on the step.

## Gate (frozen 2026-11-13, after the 680M screen, before renting)

1. **Correctness:** R's tokens identical to G's in every variant (bit-for-bit by construction).
2. **Decision (A100):** resident step at 64 positions, median of each arm's two runs. **Worth it** if R is at least 5% shorter (then a per-adapter default for large GPUs, implemented as a follow-up; the 680M keeps raw Q4_K); **Works, small gain** if 2–5%; **Rejected** if under 2% or longer.

## Protocol

One A100 SXM4 40 GB: `bench --contexts 64,1024 --resident --matvec 2,4,8 --matvec-c 16,4 --q8-fast-up-to 8 --group-matmul [--repack-q4k]`, order G1 R1 R2 G2, then one non-resident run per arm (detail profile). About 25 minutes, about $0.30.

## Log

- 2026-11-13: Added at user request ("yes" to trying the exact repack before a lossy layout), from experiment 140.
- 2026-11-13: 680M screen (`runs/experiments/DO-79/omarchy-20261113-1638-try/`): kernels `--cases decode1`: the three Q4_K matrices' outputs bit-identical with and without the repack, both within 9e-7 of NumPy; repacked one-token products slower on the 680M (memory-bound: 25.7–39.6 against 39.3–44.9 GB/s of original bytes). Full model (bench, 64 positions, grouping on): tokens identical; allocated 5.85 → 6.96 GB; GPU 87.2 → 101.5 ms (+16%). A first bench run did not apply the repack (the bench set the option after loading; fixed before the second run).
- 2026-11-13: User: "Commit, rent, $1 cap, 60 min".
- 2026-11-13 16:30–16:46 UTC: `do79` (Vast.ai A100 SXM4 40 GB, Mississippi, $0.905/h, driver 595.71.05) at 3d45f79; G1 R1 R2 G2 Gc Rc all exit 0. Logs copied back; machine deleted by recorded ID at 16:46 (an unrelated `ygaraty-panel-paid-…` machine, not made here, left alone).

## Results

| Positions | G1 | R1 | R2 | G2 | G median | R median | R against G |
|---|---|---|---|---|---|---|---|
| 64 | 17.7, 17.8 | 14.9, 14.8 | 14.6, 14.7 | 17.7, 17.6 | 17.70 | 14.75 | −16.7% |
| 1,024 | 20.5, 20.4 | 17.7, 17.2 | 17.4, 17.3 | 20.5, 20.3 | 20.45 | 17.35 | −15.2% |

Detail profile, 64 positions (profiled GPU ms), G → R: GLU (gate, up, SwiGLU) 5.14 → 3.40; DeltaNet group (qkv Q5_K, gate Q4_K, alpha, beta) 2.70 → 2.34; remaining Q4_K products 2.28 → 1.52; Q6_K 3.54 → 3.51 and Q5_K 1.48 → 1.43 (unchanged formats); norm 2.27 both. GPU step 17.2 → 14.2 ms.

Gate (by script): 1 passes (tokens identical in every variant); 2: −16.7%, at least 5%: **Worth it**.

## What it taught

- **Experiment 140's diagnosis holds:** the products were decoding-bound, not memory-bound, on the A100. Unpacking Q4_K's 6-bit scales and mins at load made its products 1.5× faster per matrix while reading 33% more bytes.
- **The trade flips with the GPU:** on the memory-bound 680M the extra bytes cost 16%; on the decode-bound A100 the removed work saves 17%. The choice must be per adapter (a large-GPU default), not global.
- **Next by the same logic:** Q6_K (3.5 ms, now the largest product family: its scales are 8-bit with an f16 super-scale, and its 6-bit values are split over two arrays) and Q5_K (qkv and ssm_out: the same 6-bit scales as Q4_K plus a high-bit plane). Both can be repacked exactly the same way.

## Limits

- One A100 host, two runs per arm.
- Only Q4_K repacked; Q5_K and Q6_K unchanged.
- Opt-in for now; the per-adapter default (choosing it on large GPUs only) and the contributor's self-test with the repacked format are the follow-up.
- Load time and memory grow (+0.9 GB here; the repack runs in JavaScript at load).

## Reproduce

```bash
# on an A100 after contributor_fleet setup, at 3d45f79
bash runs/pods-data/do79/do79.sh
```
