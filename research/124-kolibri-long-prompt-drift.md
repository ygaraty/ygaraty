# Kolibri-1's long-prompt gap is f16 cache rounding flipping near-tied experts, not a window or chunking bug

| | |
|---|---|
| **Conclusion** | Found: f16 cache rounding flips near-tied experts (1.4% jumps); f32 cache stays at 2.4e-5 (measured, layers 0–5) |
| **Status** | executed |
| **ID** | WS-41 (experiment 124) |
| **Category** | correctness |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Kolibri-1 `pipeline` against the NumPy reference: prompt lengths 19, 128, 512, 1,024; arms f16 vs f32 KV cache, the prompt in chunks of 64 vs one pass, the reference's f16 cache emulation on and off; hidden states compared layer by layer at the prompt's last position. A tiny Kolibri on the workstation first (same code paths, sliding windows shrunk to fit), then one rented A100 run for the real model |
| **Result** | The window, chunked prefill, RoPE at large positions and split combines match the reference on a window-8 tiny Kolibri (16/16 tokens, at most 1.9e-4 of the largest logit with an f16 cache, 9e-6 with f32). On Kolibri-1's layers 0–5 with a 963-token prompt in 64-token chunks and an f16 cache, 13 of 15 chunks stay at 3e-5 to 1.8e-4 of each row's largest value, but two jump to 1.4e-2 and 9.4e-4 and the next chunk is back to 1e-4; with an f32 cache every call stays at or below 2.4e-5. The reference's smallest top-6 routing margins reach 2.4e-6, so f16 rounding (about 1e-4) can push a token to a different expert. Free (workstation). |
| **Evidence** | measured-local; curated `public/research/evidence/browser/kolibri-long-prompt-drift-2026-11-12.json`; raw `runs/experiments/WS-41/omarchy-20261112-2041/` (tiny) and `omarchy-20261112-215446/` (Kolibri-1 stages); code ee2aef1 |

## Question

Experiment 122 found Kolibri-1's logits 3–15% of the largest away from the reference after a 1,002-token prompt, with the old and the new attention splits alike, from the first generated token on; experiments 113–117 only ever checked 19-token prompts. Which component causes the drift (the f16 cache, chunked prefill across window boundaries, the prefill matmul kernels, the RoPE at large positions, or the reference itself), and is it a bug?

## Prediction

A bug at a boundary rather than rounding: rounding in f16 caches gives about 1e-3 of the largest logit (experiment 113 at 19 tokens: 3.7e-4); 4–15% points at a position- or window-dependent error (sliding window boundary, chunk seams, RoPE base).

## Gate (frozen 2026-11-12, before the first run)

Diagnostic, **Found**. Workstation, a second tiny Kolibri built to exercise the window (5 layers: 4 sliding with RoPE and 1 global without, as Kolibri-1's 4-of-5 pattern; an 8-key window; 1,024 positions; vocabulary 128; random weights, Q4_K experts). `pipeline --splits 0,5 --generate 16` against the NumPy reference, prompts of 6, 16, 64 and 200 tokens (shorter and longer than the window), KV cache f16 and f32. Reported per run: tokens equal, the largest logit error relative to the largest logit, per step. A drift that appears only when the prompt passes the window points at the window path; one that grows smoothly with length and vanishes with f32 points at rounding. If the tiny model shows no drift, the real model is needed (one A100 run, budget asked first).

## Protocol

Harness flags exist for the cache precision and chunking (`--kv`, `--chunk`); a layer-by-layer comparison may need a debug output of hidden states per layer (stage boundaries via `--splits` give it at chosen layers).

## Log

- 2026-11-12: Added at user request, from experiment [122](122-attention-decode-splits-a100.md).
- 2026-11-12: started at user request ("run w-41 on workstation now"). Experiment 122 context: Kolibri-1's window is 513 keys on 4 of every 5 layers, so its 19-token gates never crossed the window; the 1,002-token prompt was the first that did. GPU kernel and NumPy reference both use "q − window < k ≤ q". Built the window-tiny model (scratch `make_tinyk2.py`).
- 2026-11-12 20:41: window-tiny sweep (`runs/experiments/WS-41/omarchy-20261112-2041/`): prompts 6, 16, 64, 200 (window 8), f16 and f32 caches: 16/16 tokens every run; largest logit error 1.2e-4 to 1.9e-4 of the largest (f16), 1e-6 to 9e-6 (f32). Then 200 tokens with 64- and 16-key attention splits (several splits combined, windows across split edges): the same. **The window, chunked prefill, RoPE at large positions and the split combine all match the reference on the tiny model.** What differs in Kolibri-1: K-quant dense weights (the tiny model's are F32) in the 64-token prefill kernels, 12 query heads per KV head (2 here), 384 experts, real activation scales. Next: Kolibri-1's first layers on the workstation (`--splits` stages compare hidden states at boundaries): the GGUF being fetched to `runs/models/kolibri-1/` (presigned URL valid until 23:35).
- 2026-11-12 21:54: GGUF fetched (44.8 GB in 4,290 s, SHA-256 510f8b5f…aec8c matches). `stages.sh` (sha256 398b1971…) on Kolibri-1 layers 0–5 as one stage against the NumPy reference, commit ee2aef1+dirty (dirty = these log files), under watch.py (stall 900 s, max 4 h); `runs/experiments/WS-41/omarchy-20261112-215446/`, finished in 8 min.
  - Short prompt (f16 cache): every call at most 3.6e-5 of the row's largest value.
  - experiment 118's long prompt (963 tokens with Kolibri's tokenizer: 15 chunks of 64, then 3 single steps), f16 cache: 13 of 15 chunks and the 3 decode steps at 3e-5 to 1.8e-4 (median 1.1e-4), but **chunk 10 jumps to 1.4e-2 and chunk 13 to 9.4e-4**, the next chunk back to 1e-4.
  - The same with an f32 cache: every call at most 2.4e-5, no jump.
  - Smallest top-6 routing margins in the reference, per layer and chunk: down to 2.4e-6, with many in 1e-6 to 1e-5 (chunk 10: 1.0e-5 at layer 1; chunk 13: 5.2e-6 at layer 1, 1.0e-5 at layer 3).
  - Reading: the error is local jumps, not a smooth drift or a boundary bug. f16 cache rounding (about 1e-4) pushes a near-tied expert choice to the other side for a few tokens, and that token's layer output then differs by percents. An f32 cache keeps the error below the margins in these 6 layers. Over 50 layers and 1,000 tokens such flips become likely, which fits experiment 122's 3–15% on the full model; the full-model f32 check needs the 49 GB plan (an A100), not run.

## Results

Tiny Kolibri (window 8, 1,024 positions, random weights), `pipeline --splits 0,5 --generate 16` against the NumPy reference:

| Prompt | f16 cache: largest logit error / largest logit | f32 cache | Tokens |
|---|---|---|---|
| 6, 16, 64, 200 | 1.2e-4 to 1.9e-4 | 1e-6 to 9e-6 | 16/16 every run |
| 200, splits 64 and 16 | same | same | 16/16 |

Kolibri-1 layers 0–5 as one stage (`stage --first 0 --last 5`), per call, the largest error relative to each row's largest value:

| Arm | Calls | Median | Largest | Calls above 5e-4 |
|---|---|---|---|---|
| 19-token prompt, f16 | 4 | 1.5e-5 | 3.6e-5 | none |
| 963 tokens in 64-token chunks, f16 | 18 | 1.1e-4 | **1.4e-2** (chunk 10) | chunk 10, chunk 13 (9.4e-4) |
| 963 tokens in 64-token chunks, f32 | 18 | 6.8e-6 | 2.4e-5 | none |

Smallest top-6 routing margin in the reference per layer and chunk: from 2.4e-6 (layer 0, chunk 5) up; chunk 10 has 1.0e-5 at layer 1, chunk 13 5.2e-6 at layer 1 and 1.0e-5 at layer 3. Not every small margin flips (chunk 5 did not): a flip needs the rounding error at that token to exceed its margin.

## What it taught

- The long-prompt failure of experiment 122 is not a bug in the window, chunking, RoPE or attention splits: all match the reference on a model built to cross the window.
- Kolibri-1's router has near-ties (margins of 1e-6 to 1e-5 among the top 6 of 384 experts) on a real prompt. f16 cache rounding is enough to swap an expert for an occasional token, and a swapped expert changes that token's layer output by percents. Errors then come in jumps, not smooth growth, and over 50 layers and 1,000 tokens such jumps become likely: this fits 3–15% at the logits.
- Consequences: a 2e-3 logit tolerance against the reference is not a valid gate for Kolibri-1 at long prompts with an f16 cache; either check with an f32 cache, compare against a reference that emulates the f16 cache exactly, or gate on token agreement and perplexity. Contributors could also offer an f32 cache (double the cache memory) where exactness matters.

## Limits

- Only layers 0–5 of 50 on the workstation (the full model needs the 49 GB plan); the full-model f32-cache check on an A100 was not run, so "explains experiment 122's 3–15%" is an inference.
- The flip itself is inferred from the per-call jump and the margins; the stage report does not record which token's expert set changed.
- One prompt.

## Reproduce

```bash
# tiny model (scratch builder in the run directory), then:
bash runs/experiments/WS-41/sweep.sh
# Kolibri-1 GGUF (44.8 GB, sha256 510f8b5f…aec8c) in runs/models/kolibri-1/, code ee2aef1 or later:
bash runs/experiments/WS-41/stages.sh runs/experiments/WS-41/<host-ts>
```
