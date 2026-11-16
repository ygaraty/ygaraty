# The 3- and 4-contributor plans for TwIL-8B and Qwen3.5-9B keep parity, fit their memory and serve

| | |
|---|---|
| **Conclusion** | **Works:** all four new plans give the reference's tokens, fit their memory and serve (measured, workstation). |
| **Status** | executed |
| **ID** | WS-13 |
| **Date** | 2026-11-09 |
| **Path** | The bundles `ygaraty model prepare --workers 2,3,4` wrote into a model root with the fixtures' memory-balanced splits: TwIL-8B 0,13,28,40 and 0,9,20,31,40; Qwen3.5-9B 0,10,24,32 and 0,6,17,28,32 (capacity 4,096; float16 cache and wire). Workstation Radeon 680M, every stage on the one GPU |
| **Result** | **Gate met for all four bundles.** Pipeline: 32/32 greedy tokens equal the reference's in each, logits within 3.8e-4 (TwIL) and 1.0e-4 (Qwen) of the largest (gate 2e-3). Serving: every request answered on relay and direct with identical texts, the 6-token limit held, and the answers equal the two-contributor bundles' word for word. Budget: largest stage allocation 2.03 GB in the 3-GiB plans and 1.62 GB in the 2-GiB plans |
| **Evidence** | Measured: [evidence/browser/three-four-worker-plans-2026-11-09.json](evidence/browser/three-four-worker-plans-2026-11-09.json) (per-bundle pipeline errors, allocations, links, answers); raw reports in `runs/experiments/WS-13/` (not in git). Bundles in `runs/model-root/` (not in git; rebuilt by `prepare --workers` from `deploy/models/*.json`) |

## Question

Do the 3- and 4-contributor plans in `twil_8b.json` and `qwen35_9b.json` hold: the same numerics with
more stage boundaries, each contributor within the plan's approved memory, and a served chat?

## Prediction

Greedy tokens unchanged except near ties (each boundary adds one float16 rounding of the hidden states);
allocations within a few percent of `ygaraty model plan`'s estimates, below `memory_gb`.

## Gate

Frozen 2026-11-09 16:41 CET, before any run. For each of the four bundles:

1. **Pipeline** (`qwen35_webgpu pipeline --splits <the bundle's> --generate 32 --kv f16 --wire f16`): every
   generated token equals the reference's greedy token, except where the reference's top-two margin is below
   0.05; logits max |error| at most 2e-3 of the largest |logit| at every step.
2. **Serving** (`scripts/user_session_check.py` with the bundle, one browser per stage, arms relay and
   direct): every request answered; the 6-token request at most 6 tokens; texts identical across the two
   arms; coherent English.
3. **Budget:** every stage's application allocation (session log `ready … allocated`) at most the plan's
   `memory_gb` GiB.

A failure of any line fails the gate for that bundle. Whether answers equal the two-worker bundles' is
reported, not gated (more float16 boundaries may flip a near tie).

## Protocol

`ygaraty model prepare --workers 2,3,4 --root runs/model-root --bundle-root <model>/v1` for both models (the
existing 2-worker bundles were rebuilt there too; every bundle's stage ranges equal its fixture plan's),
`runs/experiments/WS-13/pipelines.sh` (line 1) and `runs/experiments/WS-13/serving.sh` (lines 2 and 3; the
session driver now starts one browser per stage of the plan).

## Log

- 2026-11-09 16:41 CET: added; gate frozen; bundles prepared. Pipeline checks started.
- 16:47: pipelines passed in all four. Serving started.
- 16:52: serving passed in all four; the script deleted the browser caches.

## Results

| Bundle | Splits | Pipeline tokens | Logits error | Served (relay = direct) | Largest allocation | `memory_gb` |
|---|---|---|---|---|---|---|
| TwIL-8B, 3 workers | 0,13,28,40 | 32/32 | 3.1e-4 | yes, = 2 workers | 2.03 GB | 3 GiB |
| TwIL-8B, 4 workers | 0,9,20,31,40 | 32/32 | 3.8e-4 | yes, = 2 workers | 1.62 GB | 2 GiB |
| Qwen3.5-9B, 3 workers | 0,10,24,32 | 32/32 | 8.5e-5 | yes, = 2 workers | 2.00 GB | 3 GiB |
| Qwen3.5-9B, 4 workers | 0,6,17,28,32 | 32/32 | 1.0e-4 | yes, = 2 workers | 1.54 GB | 2 GiB |

Smallest reference margins: 0.068–0.069 (TwIL), 0.51 (Qwen), so no step relied on the near-tie exception.
Pipeline decode medians 143–151 ms per token (every stage in one browser on the 680M). Served first
requests took 5.2–6.3 s for 32 tokens with prefill; the direct route's links opened host-to-host on
loopback.

## What it taught

- The memory-balanced splits keep every stage of a plan within about 0.1 GB of the others, and the plan's
  estimate within a few percent of the measured allocation, so `memory_gb` can be set from `plan`.
- More float16 boundaries did not move a token in these prompts: all three served answers equal the
  two-contributor bundles' for both models.

## Limits

- All stages on one workstation GPU over loopback: no separate contributors or network latency.
- One served conversation per bundle (three requests); one pipeline prompt.

## Reproduce

```bash
ygaraty model prepare deploy/models/twil-8b.json --dir runs/models/twil-8b --workers 2,3,4 --root runs/model-root --bundle-root twil-8b/v1
ygaraty model prepare deploy/models/qwen3.5-9b.json --dir runs/models/qwen3.5-9b --workers 2,3,4 --root runs/model-root --bundle-root qwen3.5-9b/v1
runs/experiments/WS-13/pipelines.sh   # qwen35_webgpu pipeline --splits <bundle's> --generate 32 --kv f16 --wire f16
runs/experiments/WS-13/serving.sh     # scripts/user_session_check.py per bundle, arms R,W, one browser per stage
```
