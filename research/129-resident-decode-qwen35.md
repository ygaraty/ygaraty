# Keeping the token on the GPU now works for Qwen3.5's recurrent layers: 6.5% faster on the 680M, tokens identical

| | |
|---|---|
| **Conclusion** | Works: Qwen3.5-9B 96.2 → 90.0 ms per token on the 680M, tokens identical (measured; A100 gain unmeasured) |
| **Status** | executed |
| **ID** | WS-43 (experiment 129) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Experiment 120's resident decode (token copied to the next step's input on the GPU, read back while the next step runs, a step run past the stop rule undone) extended to stages with gated delta-net recurrent state (Qwen3.5), where undoing a step needs the state before it (a saved copy per step, or no speculative step at the end); Qwen3.5-9B; arms: current vs resident |
| **Result** | Experiment 120's resident loop extended to stages with recurrent state: the DeltaNet states are copied before every step after the first and restored when a step ran past the stop rule. On the 680M with experiment 121's kernel, B R R B arms of 32 tokens: 96.2 → 90.0 ms per token at 64 positions and 97.8 → 91.5 at 1,024 (both −6.5%), every arm's tokens identical; unit tests 5/5. Gate passes (tokens identical, not slower). The A100 gain, where about 12 ms per token is outside the GPU, is for the next rental. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/resident-decode-qwen35-680m-2026-11-13.json`; raw `runs/experiments/WS-43/omarchy-20261113-015033/`; unit tests `tests/webgpu/resident-decode.test.mjs` |

## Question

Experiment 123: our Qwen3.5-9B step on the A100 is 20.6 ms of GPU work in 32–34 ms; the resident loop (experiment 120: Kolibri 73 → 35 ms per token) is unsupported for recurrent state. Does extending it close most of that 12 ms?

## Prediction

680M: a few ms per token (GPU-bound there); A100: 32 → 22–24 ms per token (about 42–45 tokens per second), tokens identical.

## Gate (frozen 2026-11-13, before the first run)

Workstation (680M), Qwen3.5-9B, `bench --contexts 64,1024 --resident --matvec 2,4,8 --matvec-c 16,4` (experiment 120's B R R B arms per context, 32 tokens each).
1. **Correctness:** every R arm's tokens identical to the B arms' at both contexts (the arithmetic is unchanged: the step is the same plan, the snapshot a copy), and the unit tests of the snapshot path pass.
2. **Speed (680M):** R not slower than B by more than 3% at either context (the per-step snapshot copies about 50 MB of DeltaNet state, about 1 ms on the 680M).
Both: **Works** on the workstation; the per-token gain is measured on the next A100 rental (the prediction: 32 → 22–24 ms), not claimed from the 680M. Otherwise **Rejected** with the numbers.

## Protocol

`bench --resident` on Qwen3.5-9B (B R R B arms), plus unit tests for the recurrent-state rewind.

## Log

- 2026-11-12: Added at user request, from experiment [123](123-four-engines-a100.md) (WebGPU throughput improvements).

- 2026-11-13: user chose it ("WS-43 resident loop"). Code: `residentDecode` accepts stages without `residentDecode` that have `snapshot`/`restore` (Qwen3.5's DO-25 DeltaNet snapshot), snapshots before every step after the first and restores on a stop; Qwen3.5's `run` takes `gpuIds`. Unit tests 5/5 (two new: undo from the snapshot, refusal without one).

- 2026-11-13 02:0x: run `runs/experiments/WS-43/omarchy-20261113-015033/` (watch.py `finished`, stall 900 s, max 1 h), code 3d99ad2 plus the uncommitted resident-loop change.

## Results

| Context | B (normal loop) | R (resident) | R / B | Tokens |
|---|---|---|---|---|
| 64 | 96.4, 96.0 | 90.2, 89.8 | 0.936 | identical in all four arms |
| 1,024 | 98.2, 97.3 | 91.5, 91.4 | 0.936 | identical in all four arms |

ms per token, 32 tokens per arm. Gate 1 passes (tokens identical; unit tests 5/5); gate 2 passes (R 6.5% faster, the limit was 3% slower). **Works.**

## What it taught

- The snapshot (about 50 MB of DeltaNet state copied per step) costs less than the read-back wait it removes, even on the 680M where the GPU time dominates: 6 ms per token saved.
- On the A100, where experiment 123 found about 12 ms of a 34 ms step outside the GPU, the gain should be larger; experiment 120 halved Kolibri's step there this way.

## Addendum: A100 (2026-11-13, DO-73's rental)

Frozen in DO-73's gate before renting: the same bench (B R R B, 32 tokens) under Chrome and Node + Dawn on a Vast.ai A100 SXM4 40 GB (`runs/pods-data/do73/out/R129-*`).

| ms per token | Chrome B | Chrome R | Node + Dawn B | Node + Dawn R |
|---|---|---|---|---|
| 64 positions | 33.4 | **22.2** (45.1 tok/s) | 32.5 | **22.4** |
| 1,024 positions | 35.7 | **24.9** (40.2 tok/s) | 34.1 | **24.9** |

Tokens identical in every arm. On the A100 the resident loop cuts a step by 30–34%, as predicted (22–24 ms): Qwen3.5-9B at 45 tokens per second for one conversation, against 29.6 before. The step is now about the GPU time (21.9 ms at 64 positions).

## Addendum: in the production contributor (2026-11-13)

`qwen35_contributor.mjs` now runs the rest of a self-fed turn (DO-61: one stage, one conversation, no batch, no drafts, no digests) through the resident loop, counting and reporting every token by the same stop rule as before. Anything arriving mid-turn (a stop, a reset, a new prompt) ends the burst, and the newest token goes back to the front of the queue, the order the usual loop has. Default on; fixture `resident: false`, `YGARATY_RESIDENT=off` (coordinator and session harness) or `?resident=off` on the page turn it off.

Local check (`runs/experiments/resident-contributor/omarchy-20261113-020454/`), Qwen3.5-9B on the 680M, one stage, `qwen35_session local`, admission self-test passed in every run:

| | Resident off | Resident on |
|---|---|---|
| 64-token turn, median ms per token (two runs) | 97.0, 97.5 | 89.9, 90.2 |
| turn ending on its stop token (`<|im_end|>`, 8 tokens: "The capital of France is Paris.") | 98 | 90 |

Tokens identical between the arms in every case. Cancellation mid-turn (WS-28's faults) was not rerun with the resident path.

## Limits

- The A100 run (addendum) is one host, 32 tokens per arm.
- The production contributor uses it since the addendum above; cancellation and contributor-loss recovery (WS-28) not rerun with it.
- 32 tokens per arm, one run.

## Reproduce

```bash
.venv/bin/python -m ygaraty.experiments.qwen35_webgpu bench --gguf runs/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf --config runs/models/qwen3.5-9b/config.json --tokenizer runs/models/qwen3.5-9b/tokenizer.json --contexts 64,1024 --resident --matvec 2,4,8 --matvec-c 16,4 --out runs/experiments/WS-43/out
```


Ownership note (2026-11-15): public WebGPU projection; full original comparison retained in the private archive. No measurements were rerun.
