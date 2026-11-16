# Measuring the repack at load picks it correctly on the 680M but not reliably on an A100: the measurement is too short there

| | |
|---|---|
| **Conclusion** | Failed gate: the 680M chose no repack in 4 of 4 runs; the A100 chose all three formats in only 2 of 3 (measured) |
| **Status** | executed |
| **ID** | DO-83 (experiment 144) |
| **Category** | browser |
| **Where** | remote (680M part done) |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | `calibrateRepack` (webgpu_kernels): per K-quant format a synthetic 8,192 × 4,096 matrix in enough copies to exceed 64 MiB, 60 one-token products per timing cycling over the copies, raw and repacked alternating for 5 rounds; repack a format when its median is at most 0.9 of raw. Production: `kernel_defaults` for Qwen3.5 sets `repack: "auto"`; the contributor measures before loading, uses the result, reports it with `ready`; its self-test also runs every K-quant case repacked and requires the raw decoder's exact bytes. `YGARATY_REPACK=off`: never |
| **Result** | 680M (after lengthening the measurement): 4 of 4 calibrations choose nothing (repacked/raw 1.11–1.35). A100 SXM4 80 GB (three runs, the three changed files copied to the machine, hashes recorded): runs 2 and 3 choose Q4_K, Q5_K and Q6_K; run 1 chooses Q4_K and Q5_K only (Q6_K repacked/raw 0.98, over the 0.9 bar). Each A100 timing is 60 dispatches of about 0.08 ms (about 5 ms), near the browser timer's resolution. Gate: item 2 fails, **Failed gate**: `repack: "auto"` is not a production default (withdrawn before it was committed); the calibration, the `--calibrate-repack` benchmark and the self-test's byte-identity check remain, opt-in. |
| **Evidence** | measured-remote and measured-local; curated `public/research/evidence/browser/repack-calibration-2026-11-13.json`; raw `runs/pods-data/do82/out/do82/cal*` and `runs/experiments/DO-83/` |

## Question

The exact repack is −23% on an A100 (experiments 141, 142) and +25% on the Radeon 680M. Can each contributor pick it correctly by measuring its own GPU at load, in about a second?

## Gate (frozen 2026-11-13, before the A100 measurement)

1. **680M:** three calibrations choose no format (the repack is slower there in the full model).
2. **A100:** the calibration chooses Q4_K, Q5_K and Q6_K (all three were faster in the full model, experiments 141 and 142).
Both: **Works** (the production default stands). Either fails: **Failed gate** (the default is withdrawn: `repack` removed from `kernel_defaults`).

## Log

- 2026-11-13: Added and started at user request ("start 2 local"), from experiments 141 and 142.
- 2026-11-13: a production-path check through `qwen35_session local` could not run: the contributor never joins (0 tokens, timeout) both with these changes and on a clean checkout of e2c7590, so the local-session path is broken independently (it worked at 12:26 today; production-chat commits 8f83368, 482b6d9, 0f62248 changed it since). Reported, not fixed here.
- 2026-11-13: 680M (`runs/experiments/DO-83/omarchy-20261113-1953-local/`, `kernels --calibrate-repack`): with 4,096 rows and 20 dispatches the measurement was noisy (one of two runs picked Q5_K); lengthened to 8,192 rows, 60 dispatches, 5 rounds: three runs choose nothing, repacked/raw 1.30–1.35 (Q4_K), 1.11–1.16 (Q5_K), 1.19–1.30 (Q6_K). Gate item 1 passes.
- 2026-11-13 18:40–20:01 UTC: `do82` (Vast.ai A100 SXM4 80 GB, Texas, $1.422/h, driver 610.57.04) at e2c7590, shared by DO-80, DO-82 and DO-83 (this calibration after the others' runs, with `webgpu_kernels.mjs`, `qwen35_parity.mjs` and `qwen35_webgpu.py` copied in; hashes in `runs/pods-data/do82/copied-files.sha256`, equal on the machine); Kolibri fetched with a matching SHA-256; every step exit 0. Logs copied back; machine deleted by recorded ID at 20:01 (`pods ls`: 0). About $1.90 for the rental, against the $3 cap.

- 2026-11-13: the local-session path is fixed (16605e1: the general coordinator did not serve `chat_slots.mjs`, so the contributor page never started); S5-nano and Qwen3.5-9B local sessions run again (Qwen3.5-9B: 32 tokens, 89 ms per token on the 680M).

## Results

| Run | Q4_K packed/raw | Q5_K | Q6_K | Choice |
|---|---|---|---|---|
| 680M 3 | 1.33 | 1.11 | 1.19 | none |
| 680M 4 | 1.35 | 1.16 | 1.20 | none |
| 680M 5 | 1.30 | 1.11 | 1.30 | none |
| 680M 6 | 1.35 | 1.22 | 1.21 | none |
| A100 1 | 0.82 | 0.81 | 0.98 | Q4_K, Q5_K |
| A100 2 | 0.53 | 0.62 | 0.81 | all three |
| A100 3 | 0.81 | 0.79 | 0.83 | all three |

Gate (by script): 1 passes; 2 fails (run 1): **Failed gate**.

## What it taught

- The direction is right on both GPUs (no repack on the 680M; Q4_K and Q5_K always on the A100), but a fixed number of dispatches is too short on a fast GPU: 5 ms per timing is within the timer's noise. A rerun should time until each arm has run at least, say, 50 ms per round (more dispatches on faster GPUs), a new experiment with its own gate.
- The production-path check through `qwen35_session local` could not run (that path is broken independently of this change; see Log).
## Limits

- Three A100 runs on one host; the calibration was not run through a real contributor session (the local-session path is broken, see Log).

## Reproduce

```bash
.venv/bin/python -m ygaraty.experiments.qwen35_webgpu kernels --gguf ANY.gguf --config CONFIG --matvec-c 16,4 --q8-fast-up-to 8 --calibrate-repack --out OUT
```

