# Keeping the token on the GPU halves a Kolibri-1 decode step on an A100 at short context, 73 → 35 ms, same tokens

| | |
|---|---|
| **Conclusion** | Works: GPU-resident decode cuts Kolibri-1 on an A100 to 0.48× (64) and 0.72× (1k positions), tokens identical (measured) |
| **Status** | executed |
| **ID** | WS-32 (experiment 120) |
| **Category** | browser |
| **Where** | remote (user, 2026-11-12: "run ws-32 remote") |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | `residentDecode` (`inference_plan.mts`): each step's input id copied on the GPU from the previous argmax, the step submitted before the token is read back (`gpu.defer`), a step run past the stop rule rewound; stages with KV caches only (Kolibri, dense; Qwen3.5's recurrent state is excluded). Measured with `qwen35_webgpu bench --contexts 64,1024 --resident` (arms B: the usual loop, each token read back before the next step; R: resident; B R R B per context, each in a fresh slot, 32 tokens, no profiling) on Kolibri-1 on one rented A100 80 GB |
| **Result** | Works (gate by script): ms per token, B (usual loop) against R (resident): 64 positions 74.4, 72.0 against 34.8, 34.8 (ratio 0.48); 1,024 positions 135.7, 136.3 against 98.0, 97.9 (0.72); every R below every B; all 8 arms' 32 tokens identical. On this machine (A100 PCIe 80 GB) R saves about 38 ms per token at both contexts: the host work the GPU used to wait for. Kolibri at 64 positions: about 29 tokens per second in the harness. Rental $1.37 (A100 PCIe 80 GB, $0.856/h, including WS-34's reference check on the same machine; a 40 GB machine rented by mistake was deleted unused within minutes). | |
| **Evidence** | measured-remote; curated `public/research/evidence/browser/webgpu-resident-decode-a100-2026-11-12.json`; raw `runs/pods-data/ws32/` (out/bench/report.json, bench.log, steps.log); machine clone 58a6258 |

## Question

Experiment 117 left about 19 ms of host work per Kolibri token not overlapped with the GPU. With the token loop on the contributor (DO-61), nothing but the read-back needs the host between steps. Does pipelining remove the host time from the step?

## Prediction

Step ≈ max(GPU, host) instead of their sum: on the A100 47 → about 30 ms; on the 680M a smaller gain (GPU-bound).

## Gate (frozen 2026-11-12, before renting)

The bench above on the A100, Kolibri-1 (one stage of 50 layers, merged experts, host caches and fusion on), contexts 64 and 1,024, arms B R R B per context, 32 tokens each, the first step of each arm dropped.

- **Correctness:** every arm of a context gives the same 32 tokens (the B loop is the one experiments 113–117 checked against the NumPy reference).
- **Works** if R's median ms per token is at most 0.85 × B's at both contexts, with every R arm below every B arm of its context. Otherwise **Rejected**; a token difference is a **Failed gate**.
- Reported: ms per token per arm, and R against experiment 117's GPU time per token (27.9 ms).

Checked on the workstation first (tiny Kolibri, contexts 64 and 200): every arm's 32 tokens identical (21 distinct tokens); unit tests for the stop rule and rewind (`tests/webgpu/resident-decode.test.mjs`).

## Protocol

Implementation in `inference_plan.mts`/`webgpu_kernels.mts` (embedding from a token buffer, a ring of two token buffers, read-back by `mapAsync` without awaiting before the next submit); unit tests; WS-27's chat repeatability check.

## Log

- 2026-11-12: Added at user request ("to close the gap with: https://github.com/mlc-ai/web-llm - plan next set of webgpu worker optimization experiments"), in the plan [DO_webgpu_speed.md](DO_webgpu_speed.md).
- 2026-11-12: budget approved by the user: cap $4, watchdog 90 min. Rented `ws32`: the first offer was an A100 SXM4 **40 GB** (too small for Kolibri): deleted unused (`pods down ws32 --no-fetch`); then Vast.ai A100 PCIe 80 GB, South Korea, $0.856/h, watchdog 89 min. GGUF fetched by `range_fetch` (about 72 MB/s from there), SHA-256 matched. Machine clone 58a6258. `runs/pods-data/ws32/runs.sh` launched.
- 2026-11-12 13:38–13:43 (machine clock, UTC): bench exit 0 (load 231 s). Gate: **Works** (above). Then the machine ran WS-34's reference check (see experiment 121) and was deleted (`pods down ws32 --no-fetch` after copying the results); `pods ls`: only the separately approved contributor machine.

## Results

| Context | B (ms/token) | R (ms/token) | R / B |
|---|---|---|---|
| 64 | 74.4, 72.0 | 34.8, 34.8 | 0.48 |
| 1,024 | 135.7, 136.3 | 98.0, 97.9 | 0.72 |

## What it taught

- The host work per token on this A100 machine is about 38 ms (its CPU is slower than experiment 117's machine, where B took 47 ms); running the next step before reading the token back hides all of it. What remains is GPU time.
- At 1,024 positions Kolibri's step grows by about 62 ms in both arms: attention over the full-attention layers dominates at longer context on the A100, so WS-35 (flash-decoding attention) is the next lever for Kolibri.
- The rewind for a step run past the stop rule is only safe for KV-cache stages; Qwen3.5's recurrent state needs a snapshot (or a one-step delay) before the resident loop can serve it.

## Limits

- Harness only (`bench`): the contributor page still uses the usual loop; serving it in production needs the page's loop (DO-61's local loop) to call `residentDecode` and honour cancel and the turn's limit. Not done.
- A100 PCIe in Korea, not experiment 117's SXM4: absolute times differ; the arms are compared within one run.
- Two runs per arm per context; tokens compared, logits not (the B loop's logits were checked against the reference in experiments 113–117).

## Reproduce

```bash
python -m ygaraty.experiments.contributor_fleet up ws32 --gpus 1 --gpu any-80gb --min-ram 64 --disk 200 --max-hourly 2 --minutes 90
python -m ygaraty.experiments.contributor_fleet setup ws32
python -m ygaraty.experiments.contributor_fleet fetch ws32 --url-file ~/.config/ygaraty/kolibri-gguf.url \
  --dest /workspace/kolibri-1/kolibri-1-Q4_K.gguf --sha256 510f8b5f...aec8c   # full hash in experiment 113
# on the machine: runs/pods-data/ws32/runs.sh (bench --contexts 64,1024 --resident)
```
