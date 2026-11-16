# With the key/value cache preallocated, 4-bit TwIL-8B serves a 30,000-token session on four 3.5 GB-capped GPUs

| | |
|---|---|
| **Conclusion** | **Works:** the 30,000-token session that ran out of memory now completes, peak 3.0 of 3.5 GiB (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-32 |
| **Category** | correctness |
| **Where** | remote |
| **Date** | Added 2026-11-05; run 2026-11-07 |
| **Path** | [Experiment 83](083-twil-int4-four-capped-gpus.md)'s setup: TwIL-8B (revision 58d49f0d) on 4 Vast.ai containers with one 8 GB GPU each, every owner's torch allocations capped at 3,584 MiB: RTX 3070 in Nevada (coordinator, relay, w0), RTX 3060 Ti in Quebec (w1), RTX 5060 in Ontario (w2), RTX 3070 in Virginia (w3), all bf16. 10 layers per owner, `--weights int4 --head-weights int4 --dtype auto`, chunks of 64, the fused kernel, float32 wire, relay. **The fix:** each owner's key/value cache allocated once at the session's prompt plus new tokens (instead of grown in blocks by copying), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **Result** | All three pass/fail items of the frozen gate pass.<br>**Runs:** every job completed with 0 retries and 0 rejected, including the 30,000-token session with 128 new tokens (597 steps). Peak torch allocation per owner was 2,706 MiB (w0–w2) and 2,998 MiB (w3, with the head); peak reserved 2,878 and 3,124 MiB, all under the 3,584 MiB cap. nvidia-smi peaked at 3,057–3,329 MiB with the CUDA context, against 3.7–3.8 GB in experiment 83.<br>**Kernel:** float32 within 2.0–5.0e-7 of the float64 product on all four GPUs (limit 1e-5); a 4,000 MiB request was refused under the cap with expandable segments on every GPU, so the cap held.<br>**Tokens:** all 3 chat prompts' 16 tokens equal the 4-bit transformers run's.<br>**Measured:** first token after 2,950 s (49 min) at 30,000 tokens, then 1.74 s median and 2.18 s p95 per token; 1.71–1.73 s at short prompts too, so the links (100–200 ms round trips), not the context, set the token time. About $0.46 |
| **Evidence** | Curated: [evidence/correctness/twil-int4-30k-capped-memory-fixed-2026-11-07.json](evidence/correctness/twil-int4-30k-capped-memory-fixed-2026-11-07.json). Raw: `runs/pods-data/do32/` (not in git). Code: 676d6c3 (the fix), 8a28b74 (`pods` skips gone Vast offers); tests `tests/test_phi4_granite.py`, `tests/test_pods_vast.py` |

## Question

[Experiment 83](083-twil-int4-four-capped-gpus.md) ran out of memory on one owner at about 29,950 of 30,000 tokens, with 0.75 GiB of the 3.5 GiB cap reserved but fragmented. **With the cache preallocated and expandable segments, does a 30,000-token session fit under the cap on every owner?**

## Prediction

- At 30k, an owner needs about 1.1 GB of 4-bit weights (plus 0.2 GB of head on the last), 1.2 GB of cache, and about 0.5 GB of attention scratch per chunk of 64. That is about 3.0 GB, so it should fit once fragmentation is gone.
- The prefill takes about 40 min, as in experiment 83, because the links carry about 0.5 GB of activations per hop.

## Gate (frozen 2026-11-07, before any machine was made; the user's yes)

1. **Runs:** every job completes, with 0 rejected results. In particular, one session with a 30,000-token prompt and 128 new tokens completes on all owners under the cap.
2. **Kernel:** on each GPU, the fused kernel's float32 result equals the float64 product of the unpacked matrix within 1e-5 (relative). Experiment 83's limit was below bf16's own output rounding.
3. **Tokens:** the 3 chat prompts' 16 greedy tokens equal a 4-bit transformers run's (`fakequant.py`), up to its first position with a top-2 margin below 0.5.
4. **Measurement** (reported): peak torch allocation per owner (`torch.cuda.max_memory_allocated`, reported by each owner), prefill time and inter-token latency at 30k.

## Protocol

- **Code first:** preallocated `TorchLayerCache` (capacity from the session's limit), the allocator setting in `phi4-worker`, a per-owner peak-memory report, and failing a job when an owner's process dies, instead of waiting for the timeout. All committed and pushed before any machine.
- **Driver:** `runs/pods-data/do32/drive.py`, copied from do31: `up`, `kernel`, `correctness`, `long`, `down`.
- **Cost:** 4 × about $0.10/h on Vast.ai. Approved budget: $3, deleted at 2 h (first approved for Vultr; the user chose Vast.ai after Vultr's refusal).
- **Cap check:** the kernel job also asks each GPU, under the cap and `expandable_segments:True`, for 4,000 MiB; it must be refused, or the cap does not hold with expandable segments and item 1 means nothing.

## Log

- 2026-11-05: Added after experiment 83, on the user's yes.
- 2026-11-07: Gate frozen as proposed, on the user's yes, with the run moved to Vultr ("try do-32 on vultr now"; budget $3, deleted at 2 h). Code: `TorchLayerCache(capacity=)` allocated once at prompt plus new tokens (`compile_phi4_stages(capacity=)`, and the persistent loop); each owner logs `phi4 memory` (peak allocated and reserved) at a session's end; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set by the driver for every owner. Driver `runs/pods-data/do32/drive.py` (`up`, `kernel`, `correctness`, `long`, `down`) stops a job as soon as an owner's process dies.
- 2026-11-07 12:47 CET: `drive.py up` on Vultr: 2 of 4 A16s created, then the third refused ("HTTP 400: Server add failed: You have reached the maximum monthly fee limit for this account. To deploy additional services, please request a limit increase."). Both were deleted by ID within seconds, with the run's VPC; `pods ls` showed 0. Nothing was measured. Back to open, gate unchanged.
- 2026-11-07 12:50 CET: On the user's choice, moved to Vast.ai, experiment 83's setup (`SETUPS`' "4x 4 GB GPU"), same frozen gate, same $3 and 2 h. Started.
- 12:51–12:52: Vast.ai offer 59417327 was listed but renting answered `no_such_ask`, twice; the one machine made each time was deleted by ID. `pods` fixed to take the next offer (8a28b74).
- 12:53–12:58 CET: `drive.py up` (third try; see Limits for the two before). Four machines in 5 min: each fetched only its owner's tensors (3.8–4.6 GB).
- 12:59: `drive.py kernel`: gate item 2 passes on all four GPUs (2.0–5.0e-7), and the cap refuses 4,000 MiB with expandable segments.
- 13:02: `drive.py correctness`: 51 steps, 0 retries, 0 rejected; gate item 3 passes (all 3 prompts equal the 4-bit transformers run; prompt 3 differs from experiment 83's cluster at its first token, a near-tie with margin below 0.5). The driver wrongly reported "owner w0 died": workers exit when a job completes, before the coordinator does. Fixed in the driver (a worker that is gone counts as a death only if the job has no result 20 s later).
- 13:02–13:56: `drive.py long`: the 30,000-token session completes, 597 steps, 0 retries, 0 rejected; gate item 1 passes.
- 13:57: `pods down do32 --no-fetch`: all four deleted by recorded ID; `pods ls` showed no do32 machines (DO-39's pool, run in parallel, was left alone).

## Results

| Gate item | Outcome |
|---|---|
| 1. Runs | **Pass.** Kernel, correctness and 30,000-token jobs completed, 0 retries, 0 rejected. The 30,000-token session completed on all owners under the cap. |
| 2. Kernel ≤ 1e-5 (float32 vs float64) | **Pass:** 2.0–5.0e-7 on every GPU and matrix (qkv, o, gate_up, down, head). |
| 3. Tokens vs the 4-bit transformers run | **Pass:** 3 of 3 prompts equal for all 16 tokens. |
| 4. Measurement | Below. |

Peak memory in the 30,000-token session (MiB; cap 3,584):

| Owner | GPU | Torch allocated | Torch reserved | nvidia-smi (with context) |
|---|---|---|---|---|
| w0 | RTX 3070 | 2,706 | 2,878 | 3,083 |
| w1 | RTX 3060 Ti | 2,706 | 2,878 | 3,057 |
| w2 | RTX 5060 | 2,706 | 2,878 | 3,087 |
| w3 (head) | RTX 3070 | 2,998 | 3,124 | 3,329 |

Reserved minus allocated is 126–172 MiB, against experiment 83's 0.75 GiB of fragmentation; about 460 MiB of the cap is left on the head owner.

Timing:

| Prompt | First token | Per token (median) | p95 |
|---|---|---|---|
| 3 chat prompts (~20 tokens) | 4.9–46.3 s (the first includes warmup) | 1.71–1.73 s | |
| 30,000 tokens | 2,950 s | 1.74 s | 2.18 s |

## What it taught

- **The out-of-memory in experiment 83 was fragmentation, not size.** Allocating the cache once at its final size, with expandable segments, brings the peak to 2.7–3.0 GiB allocated, as predicted (about 3.0 GB).
- **The cap still holds with expandable segments.** `set_per_process_memory_fraction` refused 4,000 MiB on every GPU with `expandable_segments:True`.
- **Per-token time at 30k is set by the links, not the context.** 1.74 s at 30,000 tokens is the same as at 20 tokens (1.71–1.73 s). It is slower than experiment 83's 0.95 s because this pool's hosts were farther apart (100–200 ms round trips against 57–88 ms).
- **Prefill took 49 min,** longer than experiment 83's ~40 min, again from the longer links: 469 chunks of 64 tokens, each one a full pass through four owners.
- **Bugs found:** Vast.ai's search can list an offer that renting calls gone (`no_such_ask`); `pods` now tries the next offer (8a28b74). The driver's owner-death check confused a finished job with a crash; fixed before the long job.

## Limits

- One 30,000-token session, one seed (random token IDs, so its 128 tokens are newlines and carry no correctness signal; tokens are checked on the 3 chat prompts).
- All four GPUs were Ampere or newer (bf16); experiment 83 had a float16 RTX 2070, which this run did not cover.
- The 3.5 GB cap emulates a 4 GB card; a real 4 GB card's driver and context may leave a different margin (here about 460 MiB on the head owner).
- Two attempts did not start: on Vultr (4 A16s in one VPC, $1.88/h) the third machine was refused at the account's monthly fee limit, and on Vast.ai offer 59417327 was listed but gone, twice. The machines each made existed for seconds and were deleted by ID.

## Reproduce

```bash
.venv/bin/python runs/pods-data/do31/reference.py        # needs runs/models/twil-8b (17.6 GB); reference.json
.venv/bin/python runs/pods-data/do31/fakequant.py        # fakequant.json (4-bit transformers tokens)
.venv/bin/python runs/pods-data/do32/drive.py up          # 4 Vast.ai 8 GB machines, ~$0.43/h, deleted at 2 h or $3
.venv/bin/python runs/pods-data/do32/drive.py kernel
.venv/bin/python runs/pods-data/do32/drive.py correctness
.venv/bin/python runs/pods-data/do32/drive.py long
.venv/bin/python runs/pods-data/do32/drive.py down
```
