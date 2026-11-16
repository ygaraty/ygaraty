# A 32-conversation decode step is arithmetic-bound in the matrix products, not in per-conversation work

| | |
|---|---|
| **Conclusion** | Found: matrix products are 87% of a 32-conversation step, flat at about 720 GFLOP/s: arithmetic-bound (measured, 680M) |
| **Status** | executed |
| **ID** | WS-44 (experiment 126) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Qwen3.5-9B on the Radeon 680M, `bench --batches 1,8,32 --batch-profile` with experiment 125's kernel (`--matvec 2,4,8 --matvec-c 16,4 --matvec-b 8`): GPU time per dispatch name, dispatch counts and host time per step |
| **Result** | Qwen3.5-9B with experiment 125's batched kernel on the Radeon 680M, GPU ms per step at 1 / 8 / 32 conversations: 88.5 / 237.0 / 908.2, of which the matrix products 82.6 / 202.9 / 787.8 (93 / 86 / 87%) and the delta-net update 2.4 / 20.5 / 88.6 (3 / 9 / 10%); everything else under 4%. From 8 to 32, each conversation adds 24.4 ms of matrix products, 2.8 of delta and 1.2 of host encoding (dispatches 518 / 1,365 / 4,269). The products run at 217 / 706 / 727 GFLOP/s: flat from 8 on, so they are arithmetic-bound, not memory-bound. This corrects experiment 125's reading that per-conversation work set its plateau. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/batch-step-profile-680m-2026-11-12.json`; raw `runs/experiments/WS-44/omarchy-20261112-234730/` (report, watch log); code 903d58f + `--batch-profile` |

## Question

Experiment 125: past 8 conversations each one adds about 9 ms per step on the A100 in both batch paths, so the matrix products are no longer the limit. Which operations grow with the number of conversations (attention, delta-net state updates, per-slot dispatches, the head and argmax, host encoding)?

## Prediction

Per-slot attention and delta-net dispatches (one per conversation per layer) dominate at 32, with host encoding growing linearly with the dispatch count.

## Gate (frozen 2026-11-12, before the run)

Diagnostic, **Found**: per batch size, GPU ms per step by operation, dispatches per step, encode ms; the operations whose time grows with the batch, ranked by their share of the step at 32. No pass/fail; the 680M's arithmetic limit (experiment 125's screen) is noted where it shapes the matrix-product share.

## Protocol

One run; detail profiling adds a timestamp per dispatch (absolute times inflate; shares and growth are the evidence).

## Log

- 2026-11-12: Added at user request, from experiment [125](125-batched-matvec-a100.md). `--batch-profile` added to the bench.
- 2026-11-12 23:5x: started, code 903d58f+dirty (`--batch-profile`), `bench --batches 1,8,32 --batch-profile --matvec 2,4,8 --matvec-c 16,4 --matvec-b 8` into `runs/experiments/WS-44/omarchy-20261112-234730/`, watch.py stall 900 s, max 1 h.

- 2026-11-12 23:52: finished (watch.py `finished`).

## Results

| Conversations | Wall ms per step | GPU ms | Matrix products | Delta-net | Encode ms | Dispatches | Products GFLOP/s |
|---|---|---|---|---|---|---|---|
| 1 | 98.8 | 88.5 | 82.6 (93%) | 2.4 | 2.3 | 518 | 217 |
| 8 | 265.0 | 237.0 | 202.9 (86%) | 20.5 | 7.2 | 1,365 | 706 |
| 32 | 1,018.3 | 908.2 | 787.8 (87%) | 88.6 | 36.4 | 4,269 | 727 |

GFLOP/s: 2 × 8.95 billion parameters × conversations ÷ the products' GPU time.

## What it taught

- The batch plateau is the matrix products running out of arithmetic, not per-conversation dispatches: from 8 conversations on, their throughput stays at about 720 GFLOP/s, a fifth of the 680M's float32 peak (about 3.4 TFLOP/s), because every weight is dequantized again in each pass of 8 and each product is a scalar float32 FMA.
- **Correction to experiment 125:** its "past 8 the matrix products are no longer the limit" was an inference from both paths growing alike; this profile shows both are arithmetic-bound in the products. On the A100 the same reading fits: 32 conversations in 299 ms is about 2 TFLOP/s of products, about a tenth of its float32 peak.
- What would help: decode each weight block once for all 32 conversations (passes of 16 or 32, in registers or workgroup memory), and tensor-core products (`subgroup-matrix`, exposed on the A100 in Chrome, experiment 123); the delta-net update (10%) and host encoding (1.2 ms per conversation) come next.

## Limits

- 680M only (arithmetic-poor next to the A100); the A100 share is an estimate from experiment 125's timings, not a profile.
- Detail profiling inflates absolute times (wall 98.8 against 97 without it); shares and growth are the evidence. One run.

## Reproduce

```bash
.venv/bin/python -m ygaraty.experiments.qwen35_webgpu bench --gguf runs/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf --config runs/models/qwen3.5-9b/config.json --tokenizer runs/models/qwen3.5-9b/tokenizer.json --batches 1,8,32 --batch-profile --matvec 2,4,8 --matvec-c 16,4 --matvec-b 8 --out runs/experiments/WS-44/out
```
