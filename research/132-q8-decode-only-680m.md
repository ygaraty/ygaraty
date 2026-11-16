# Used for decode-sized calls only, the Q8_0 decoder is 1.2–2.8× faster and never slower: now the default

| | |
|---|---|
| **Conclusion** | Works, now default: Q8_0 products 1.2–2.8× faster at 1–8 tokens, unchanged at 9+, exact (measured, 680M) |
| **Status** | executed |
| **ID** | WS-47 (experiment 132) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | med |
| **Date** | Added 2026-11-13 |
| **Path** | Experiment 130's Q8_0 decoder used only for calls of at most 8 tokens (`gpu.q8FastUpTo = 8`, `--q8-fast-up-to 8`): one-token decode (coalesced kernel) and decode batches (batched kernel); prompt chunks keep the generic kernel; Qwen3.5-9B on the 680M; arms G (generic, the default) and D (decode-only) |
| **Result** | Kernel cases, medians of two runs per arm (G generic, D decode-only): ssm_alpha 0.097 → 0.066 ms at 1 token, 0.180 → 0.083 at 8; ssm_beta 0.074 → 0.061 at 1, 0.179 → 0.065 at 8; at 9, 32 and 64 tokens D/G 0.93–1.011 (the same generic kernel). Every D case within 2.8e-7 of float64; base-64 tokens identical in all four runs. Gate passes (no case above 1.05× G). **Works**: `q8_fast_up_to = 8` is now in the session defaults (`YGARATY_KERNELS=off` switches back). Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/q8-decode-only-680m-2026-11-13.json`; raw `runs/experiments/WS-47/omarchy-20261113-023423/` |

## Question

Experiment 130 rejected the decoder at its frozen 3× bar (2.3× on decode) and found it slower for prompt chunks (9+ tokens). Used for decode-sized calls only, is it at least as fast as the generic kernel at every call size, with the same tokens? (User, 2026-11-13: "isn't 2.3x enough to keep it?", then "ok" to this check.)

## Prediction

Faster at 1–8 tokens (1.4–4.4× in experiment 130's cases), identical at 9+ (the same generic kernel).

## Gate (frozen 2026-11-13, before the first run)

Order G D D G. `kernels --cases q8 --repeat 50` (ssm_alpha and ssm_beta at 1, 2, 3, 8, 9, 32, 64 tokens) and `bench --contexts 64` per run, both with `--matvec 2,4,8 --matvec-c 16,4 --matvec-b 8`.
1. **Correctness:** every D kernel case within 1e-5 of float64 relative to the largest output; base-64 tokens identical in all four runs.
2. **Never slower:** for every case, D's median ms at most 1.05× G's (the 5% allows timing noise; at 9+ tokens both run the same kernel).
Both: **Works**, and `q8_fast_up_to = 8` joins the session defaults (`kernel_defaults`, switch `YGARATY_KERNELS=off`). Otherwise **Rejected** with the numbers.

## Protocol

Driver `runs/experiments/WS-47/run.sh`.

## Log

- 2026-11-13: Added and started at user request; code: `gpu.q8FastUpTo` / `--q8-fast-up-to`.

- 2026-11-13 02:37: `runs/experiments/WS-47/run.sh` into `runs/experiments/WS-47/omarchy-20261113-023423/` (watch.py `finished`, 3 min). Gate passes; `q8_fast_up_to=8` added to `KERNELS_DEFAULT`, applied by the contributor and its self-test.

## Limits

- 680M only; on the A100, where the generic Q8_0 products took 2.98 ms per token (experiment 123), the gain is unmeasured.
- Kernel timings at 9+ tokens include run-to-run noise of up to 7% (0.93 at 9 tokens for the same kernel).

## Reproduce

```bash
bash runs/experiments/WS-47/run.sh runs/experiments/WS-47/<host-ts>
```
