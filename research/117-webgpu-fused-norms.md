# Fusing sandwich norms with their residual adds and the QK-norm pair cuts a Kolibri-1 token 52 → 47 ms on one A100, bitwise identical

| | |
|---|---|
| **Conclusion** | Works: fused norms cut a Kolibri-1 token 52.3 → 47.2 ms on one A100, bitwise identical (measured) |
| **Status** | executed |
| **ID** | DO-66 (experiment 117) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | The DO-63 machine (Vast.ai A100 80 GB) after DO-65; Kolibri-1, one stage of 50 layers (`pipeline --splits 0,50 --generate 32 --profile`, merged experts, host caches on); arms F (fusion: the plan executor runs each sandwich norm with its residual add as one dispatch, `norm_add`, and the query and key norms as one, `norm_pair`) and U (`--no-fuse`) |
| **Result** | Works (gate by script): F's GPU time below U's in both pairs (27.8 < 28.2, 27.9 < 28.2 ms); median step 47.2 (F) against 52.3 ms (U), −10%; dispatches 1238 against 1388 (150 fused: 100 `norm_add`, 50 `norm_pair`); JavaScript encode 7.6 against 10.1 ms, plan expansion 3.8 against 4.8 ms, submission wall 43.1 against 47.3 ms. All 4 runs valid (32/32 tokens, logits within 2.8e-4); logits bitwise identical to F1's (32/32 files). About 21 tokens per second in this harness. The saving is mostly host time (fewer dispatches to encode), not GPU time (−0.35 ms). |
| **Evidence** | `runs/pods-data/do66/` (gate.json, F1 U2 U3 F4 reports, logs, logits, files.sha256: webgpu_kernels.mjs 4240eb92…, inference_plan.mjs f3ad031f…, qwen35_parity.mjs 8ee918fb…, qwen35_webgpu.py db893a9b…); curated: `public/research/evidence/browser/kolibri-a100-decode-speed-2026-11-12.json`; code committed as 430c583 |

## Question

[DO-64](115-kolibri-a100-operation-bench.md) (experiment 115) costed the small per-layer operations every layer-only worker runs: six RMS norms (0.014–0.020 ms each) and two residual adds per layer. Fusing a norm with the add that consumes it, and the query norm with the key norm, removes three dispatches per layer (150 of about 1,388 per Kolibri token) without changing the arithmetic. How much time does it save on the A100, with tokens and logits unchanged?

## Prediction

GPU time falls by roughly the removed dispatches' cost (about 3 × 0.015 ms per layer ≈ 2–3 ms per token) and the submission's wall time by a little more (fewer dispatches to encode and schedule); logits bitwise identical (tiny Kolibri: 16/16 files).

## Gate (frozen 2026-11-12, before running)

- Runs F U U F, each `pipeline --splits 0,50 --generate 32 --kv f16 --wire f16 --profile` (merged experts, host caches). Valid: 32/32 tokens equal to the reference, logits within 2e-3 of the largest.
- **Correctness:** every run's logits bitwise identical to the first F run's. A break is a **Failed gate**.
- **Works** if F's median GPU ms per token is below U's in both pairs and F's median step time is at most U's; otherwise **Rejected**. The size of the saving is reported, not gated.

## Protocol

- Code: the machine's clone plus hand-copied `webgpu_kernels.mjs`, `inference_plan.mjs`, `qwen35_parity.mjs`, `qwen35_webgpu.py` (hashes in `/workspace/do66/files.sha256`), copied after DO-65's last run. Checked on the workstation: tiny Kolibri, fused and unfused 16/16 tokens equal to the reference, logits bitwise identical (16/16), dispatches 86 → 77 per token; WS-27's S5-nano chat sessions (1 and 2 workers) bitwise identical to WS-27's run 1; WebGPU tests 132/134 (the 2 failures also fail on clean HEAD), Python tests 42 passed.
- Budget: user, 2026-11-12: "work on 1, 2 and 3. budget approved"; within the running rental (watchdog 05:38).

## Log

- 2026-11-12: Added at user request (item 2 of DO-63's findings: fusion; "such as QK-Norm, attention gating, sandwich norm"). Attention gating is already fused into the norm kernel (`use_gate`, Qwen3.5's gated norm); Kolibri has none. Gate frozen before running.
- 2026-11-12 04:21–05:05 CET: runs F1 U2 U3 F4 (`do66.sh`, each exit 0). Gate: **Works** as in Result. 05:06: `pods down do63 --no-fetch`; `pods ls` 0 machines. The rental (DO-63 to DO-66): 2 h 27 min × $1.634/h = **$4.01** (cap $6).
- 2026-11-12: written up as experiment 117.

## Limits

- One machine, one prompt, 32 tokens; two runs per arm. F4's step (50.1 ms) is close to U's (52.3); the GPU gate holds in both pairs but the GPU saving is small (0.35 ms).
- The saving is mostly host time (encode 10.1 → 7.6 ms): on a faster CPU or with fewer dispatches elsewhere it shrinks.
- Fusion is on by default (`gpu.fuse`); `--no-fuse` switches back. It is a peephole in `executePlan`: it applies to any family whose plan has `rms_norm` then `add`, or two consecutive grouped `rms_norm`s; checked bitwise on Kolibri (tiny and full) and WS-27's S5-nano sessions only.
- Attention gating was already fused (Qwen3.5's gated norm); Kolibri has none.

## Reproduce

```bash
# rented 80 GB GPU with the Kolibri-1 GGUF, code at 430c583 or later
bash runs/pods-data/do66/runs.sh   # F U U F: pipeline --splits 0,50 --generate 32 --kv f16 --wire f16 --profile [--no-fuse]
```
