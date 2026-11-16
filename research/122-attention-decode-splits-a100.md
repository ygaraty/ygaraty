# Smaller attention splits make Kolibri-1 decode 3.6× faster at 1k context on an A100, but the long-prompt reference check fails in both arms

| | |
|---|---|
| **Conclusion** | Failed gate: attention 6.9× faster at 1k positions, but the 1,000-token reference check fails, old splits too (measured) |
| **Status** | executed |
| **ID** | WS-35 (experiment 122) |
| **Category** | browser |
| **Where** | workstation, then the user's rented A100 (contrib1) |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Decode attention split over positions across workgroups (partial max, sum and output per split, subgroup reductions, combined in a fixed order), for Qwen3.5's full-attention layers and Kolibri's windowed and full layers; arms: current vs split; contexts 64, 1k, 4k |
| **Result** | Speed passes: attention GPU ms per token (attention and combine) 94.4 → 13.8 at 1,024 positions (6.9×), 94.2 → 19.2 at 4,096 (4.9×), 19.3 → 13.2 at 64; with experiment 120's loop Kolibri-1 runs at 30.5 instead of 111.0 ms per token at 1,024 positions (32.8 against 9.0 tokens per second) and 29.9 instead of 36.1 at 64; every arm's tokens identical. Correctness fails as frozen: with a 1,002-token prompt, 30/32 tokens and logits up to 2.0e-1 of the largest (limit 2e-3). The control with the old splits fails the same way (31/32, up to 1.5e-1), with the error already at the first token (the prompt pass): the gap is in long prompts, never checked before (experiments 113–117 used 19 tokens), not in the splits. Rental: contrib1 (shared with DO-61 part 2 and the queued Kolibri runs), $6.99 in all. |
| **Evidence** | measured-remote; curated `public/research/evidence/browser/webgpu-attention-splits-a100-2026-11-12.json`; raw `runs/pods-data/contrib1/ws35/` (roofline, resident and pipeline reports and logs) and `runs/pods-data/contrib1/k2/` (the profile before); code c44ed00 |

## Question

Attention was 24% of Kolibri's GPU time at about 40 positions (0.21 ms per layer), far above its few kilobytes of cache reads, and grows with context. Does a split kernel make it launch-bound at short context and bandwidth-bound at long context?

## Prediction

Short context under 0.03 ms per layer equivalent; at 4k within 2× of the cache-read time.

## Gate (frozen 2026-11-12, before the measured runs)

First step of the plan (not yet the full flash-decoding rewrite): decode attention in splits of 64 keys instead of 512, so a long context is spread over many workgroups (`gpu.attentionSplit`, `--attention-split 64`; the kernel and the combine are unchanged). Measured on Kolibri-1, one stage, on the rented A100 SXM4 80 GB (`contrib1`), with experiment 121's kernel and 120's loop on.

1. **Speed:** `bench --contexts 64,1024,4096` (detail profile) with split 512, then with split 64; attention GPU ms per token (`dense_attention*` + `dense_combine*`) and step ms per context. Also `bench --contexts 64,1024 --resident` per arm for ms per token.
2. **Correctness:** `pipeline --splits 0,50` on Kolibri-1 with split 64 and a prompt of about 1,000 tokens (so the window spans many splits), 32 generated: 32/32 tokens equal to the NumPy reference, logits within 2e-3 of the largest.

**Works** if attention GPU ms per token falls at least 3× at 1,024 and at 4,096 positions, with correctness. At 64 positions the change does nothing (one split either way); reported, not gated. Otherwise **Rejected**; a correctness break is a **Failed gate**.

## Protocol

Kernel in `webgpu_kernels.mts`; the split count chosen from the context length; tests on the tiny models.

## Log

- 2026-11-12: Added at user request ("to close the gap with: https://github.com/mlc-ai/web-llm - plan next set of webgpu worker optimization experiments"), in the plan [DO_webgpu_speed.md](DO_webgpu_speed.md).
- 2026-11-12: experiment 120's A100 profile (contrib1, Kolibri-1, new kernel): attention 18.6 ms GPU per token at 64 positions, 93.4 at 1,024 and 93.3 at 4,096 (Kolibri's window), 83% of GPU time; the kernel launches only ceil(positions / 512) splits × 4 KV heads workgroups. Implemented the decode split (opt-in); tiny Kolibri: tokens identical between split 512 and 64 at 64 and 200 positions (all 8 arms). User: "extend gpu time if necessary": contrib1's deadline moved to 20:26 local.
- 2026-11-12 18:28–19:12 (machine clock, UTC 17:2x–18:0x): `runs/pods-data/contrib1/ws35.sh` on contrib1 (files copied: dense_gpu.mjs 5551186a…). Speed (bench, new kernel, split 512 → 64): attention GPU ms per token 93.4 → 12.6 at 1,024 positions (7.4×), → 16.8 at 4,096; resident loop 111.0 → 30.5 ms per token at 1,024 (9.0 → 32.8 tokens per second), 36.1 → 29.9 at 64; every arm's tokens identical.
- 19:30 (local): correctness run with split 64 and a 1,002-token prompt: **30/32 tokens, logits up to 8e-2 of the largest** (limit 2e-3). But step 0 (the prompt pass, which the decode split does not touch: prefill chunks keep 512) already differs by 4.3e-2, and the two differing tokens are near-ties (top-2 margins 0.014 and 0.053): the gap comes from the long prompt, which experiments 113–117 never checked (their prompts had 19 tokens). Control launched: the same run with split 512. contrib1's deadline moved to 21:11 local (at most $7.81 in all), under the user's "extend gpu time if necessary".
- 2026-11-12 20:13 (local): control (split 512, same prompt): 31/32 tokens, logits 2.8e-2 to 1.5e-1 of the largest, step 0 at 3.3e-2. Both arms fail by the same margin from the prompt pass on: the long-prompt gap predates this change. (Step 0 differs a little between arms because the first prompt chunks also fit the 64-key splits.) Results copied back; contrib1 deleted (`pods ls`: 0); $6.99 for the whole rental.

## Results

| Kolibri-1 on the A100 | Split 512 | Split 64 |
|---|---|---|
| Attention GPU ms per token, 64 positions | 19.3 | 13.2 |
| Attention GPU ms per token, 1,024 positions | 94.4 | 13.8 |
| Attention GPU ms per token, 4,096 positions | 94.2 | 19.2 |
| Resident loop, ms per token at 64 | 36.1 | 29.9 |
| Resident loop, ms per token at 1,024 | 111.0 | 30.5 |
| Reference check, 1,002-token prompt: tokens | 31/32 | 30/32 |
| Reference check: largest logit error / largest logit | 1.5e-1 | 2.0e-1 |

## What it taught

- The windowed attention kernel was starved of parallelism, not slow per key: 512-key splits gave a 1,024-key window about a dozen workgroups on a 108-SM GPU. 64-key splits alone remove 85% of its time; Kolibri-1 now decodes at about 33 tokens per second at any context in the harness, from 9 at 1k this morning.
- Kolibri-1's long-prompt numerics drift from the NumPy reference by a few percent of the largest logit (f16 cache, chunked prefill, or the reference's own emulation of them): a separate question, needed before any long-context correctness gate.
- Freezing a correctness gate on a check never run before (long prompts) made this a Failed gate for a cause outside the change; a control arm in the gate itself would have isolated it.

## Limits

- The split is opt-in (`--attention-split 64`); not default, since the gate failed. A rerun with a short-prompt reference check plus the long-prompt comparison against split 512 is a new experiment.
- Profile timings carry experiment 120's caveat (base and base-again 7% apart in the earlier profile); the resident runs are the timing evidence.
- One machine; Qwen3.5's attention kernel (a different one) unchanged.

## Reproduce

```bash
# rented 80 GB GPU with the Kolibri-1 GGUF (see experiment 120), code at c44ed00 or later:
bash runs/pods-data/contrib1/ws35.sh   # roofline and resident benches at splits 512 and 64; pipeline at 64
# control: the same pipeline with --attention-split 512
```
