# 4-bit TwIL-8B runs on four GPUs capped at 3.5 GB, at 0.95 s per token, but runs out of memory near 30k tokens

| | |
|---|---|
| **Conclusion** | **Failed gate:** TwIL-8B served by four 3.5 GB-capped GPUs at 0.95 s/token, but out of memory near 30k tokens (measured). |
| **Status** | executed |
| **ID** | DO-31 |
| **Date** | 2026-11-05 |
| **Path** | [webAI-Official/TwIL-8B](https://huggingface.co/webAI-Official/TwIL-8B) (Granite 4.2, 8.8B; "balanced" variant at revision 58d49f0d), read as Granite by `phi4_checkpoint`, through the Phi-4 pipeline.<br>**Machines:** 4 Vast.ai containers with one 8 GB GPU each: RTX 4060 in British Columbia (coordinator, relay, w0), RTX 2070 in Ontario (w1, float16), RTX 3070 in Nevada (w2), RTX 3070 in Quebec (w3). Each owner's torch allocations were capped at 3,584 MiB to stand in for a 4 GB card.<br>**Settings:** 10 layers per owner, `--weights int4 --head-weights int4 --dtype auto`, embedding in host memory, prefill chunks of 64, the fused 4-bit decode kernel (arm A) or matrices unpacked (arm B), float32 wire, relay and handoffs. Each machine downloaded only its own tensors by HTTP range |
| **Result** | Three of the four frozen gate items fail; the timings are real.<br>**Runs:** every job completed with 0 retries and 0 rejected, except the 30,000-token session. There w3 ran out of memory in attention at about 29,950 cached tokens: 2.57 GiB was allocated and 0.75 GiB reserved but fragmented.<br>**Kernel:** within 1e-3 only on the float16 GPU; bf16 output rounding alone is up to 3.9e-3, and the kernel's float32 result is within 3e-7.<br>**Tokens:** the first token equals the bf16 reference on 1 of 3 prompts, and a 4-bit transformers run gives the cluster's tokens.<br>**Measured:** 0.95 s per token with the fused kernel against 1.41 s unpacked, on 57–88 ms links. Cost about $1.03 of $2 |
| **Evidence** | Measured: [evidence/correctness/do31-twil-int4-4gb-2026-11-05.json](evidence/correctness/do31-twil-int4-4gb-2026-11-05.json); raw logs in `runs/pods-data/do31/`. Code 1a5c96c, 72a5fc7, e7bc569 (phi4_owner.py hand-copied, sha256 e0b0c748…). Tests `tests/test_phi4_granite.py`, `tests/test_phi4_fetch.py` |

## Question

**Can 4 GB contributors serve TwIL-8B, at the 32k context its reasoning needs, and at what latency?** No 4 GB GPUs are for rent (Vast.ai listed none in the US or Canada on 2026-11-05), so 8 GB cards with a 3.5 GB allocation cap stood in for them. It was also the first live run of the Granite loader, the fused 4-bit kernel, float16 on a pre-Ampere GPU and per-tensor range downloads.

## Prediction

- **Memory:** an owner holds 10 layers in 4-bit (1.06 GB), at most the 4-bit head (0.22 GB), and 4 KB of keys and values per token per layer (1.34 GB at 32k), plus scratch. That is about 2.8 GB at 32k, under the 3.5 GB cap.
- **Compute:** the fused kernel reads only the packed codes, so a few ms per owner per token if bandwidth-bound; with launch and Python overhead, 10–30 ms.
- **Network:** 4 owners anywhere in the US and Canada, each hop through the relay on machine 0; experiment 82 measured about 75–85 ms per relay call.
- **Inter-token latency:** 0.3–0.8 s with the fused kernel, about 1.5 s or more unpacked.

## Gate (frozen 2026-11-05, before any machine was made)

1. **Runs:**
   - Every job completes, with 0 rejected results.
   - No owner fails an allocation under the 3.5 GB cap, including in one session with a 30,000-token prompt and 128 new tokens.
2. **Kernel:** on each GPU, the fused 4-bit product equals the unpacked one within a relative error of 1e-3.
3. **Tokens:**
   - Inputs: 3 chat prompts with 16 greedy tokens each, compared against a bf16 transformers reference computed on the workstation.
   - Pass if every prompt's first token equals the reference's.
   - Reported: how many tokens match before the first divergence, and the reference's top-2 margin there.
4. **Measurement** (reported, not pass/fail):
   - Arms: fused kernel on (A), and `YGARATY_INT4_FUSED=0` (B), one job per arm and length (A B, B A, A B).
   - Sessions: prompts of 128, 1,024 and 8,192 tokens (seed 2026), 128 new tokens, 1 warmup plus 3 measured.
   - Reported: time to first token, median and p95 inter-token latency, peak GPU memory, and each host's location and round trip to machine 0.

## Protocol

- **Reference:**
  - Done on the workstation, at no cost: `runs/pods-data/do31/reference.py`.
  - The checkpoint (17.6 GB) was downloaded to `runs/models/twil-8b`.
  - Transformers 5.22.1 ran in bf16 on the CPU, recording tokens and top-2 margins.
- **Machines:** `runs/pods-data/do31/drive.py up`.
  - It makes, ships and sets up the 4 machines through `phi4_cluster.Cluster` (setup "4x 4 GB GPU", `any-8to9gb-sm75` in US and CA).
  - The watchdog deletes them at 180 min.
- **Jobs:** `drive.py kernel`, `rtt`, `timing 128`, `correctness`, `long`. Each job starts the coordinator on machine 0 and one owner per machine (`YGARATY_MODEL=twil YGARATY_GPU_MEMORY_MB=3584`), then fetches every machine's run folder.
- **Diagnostic:** `runs/pods-data/do31/fakequant.py` (workstation).
  - It replaces every layer matrix and the head of transformers' model with `dequantize(quantize(w))`.
  - It then generates for the same prompts, to separate quantization from pipeline errors.
- **Teardown:** `pods down do31 --no-fetch`, then `pods ls` showed 0.

## Log

- 2026-11-05: Added. The user asked for TwIL-8B on 4 GB contributors, starting with the smaller GPUs before "real" ones.
  - Code: Granite loader, fused 4-bit kernel, float16 path, 4-bit head, host embedding, growing KV cache, range downloads, and `phi4_chat --model twil`.
  - 62 local tests pass; the kernel was checked in Triton's interpreter only.
- 2026-11-05: The user approved a $2 budget with deletion at 3 h, froze the gate as proposed, and approved the local bf16 reference.
- 10:06–10:11: `drive.py up` at commit 72a5fc7: 4 machines at $0.402/h, set up in 5 min.
  - Range downloads took 3.8–4.5 GB per machine in 19–87 s. The bf16 checkpoint holds 10 layers in about 4 GB; the planned "1–2 GB" was the 4-bit size, which is quantized on the machine.
- 10:11: `drive.py kernel`. **Gate item 2 fails as written** on the 3 bf16 GPUs: 1.9–3.2e-3 against 1e-3. On the float16 RTX 2070 it is 2.4–4.3e-4.
  - Diagnostic (not a rescue of the gate): the kernel's float32 result is within 1.9–2.8e-7 of exact. Its bf16 error equals the exact product rounded to bf16.
- 10:13: The first 30k job failed every load: `cap_gpu_memory` passed "cuda" without a device index.
  - Fixed in e7bc569 and copied to the 4 machines by hand (phi4_owner.py sha256 e0b0c748…).
  - A manual load of w0 then took 19 s: 1,016 MiB allocated, 1,596 MiB peak.
- 10:16–10:22: The 30k job reran and was stopped after 25 of 469 chunks, at about 13 s per chunk.
  - The 1,024 and 8,192-token timings no longer fit the 3 h window. They were dropped, and the remaining jobs reordered.
- 10:23: `drive.py rtt`: Ontario 72.9 ms, Nevada 57.2 ms, Quebec 88.4 ms (TCP connect to machine 0, median of 10).
- 10:23–10:48: 128-token timing, A then B: 520 steps each, 0 retries, 0 rejected, the same tokens in every session.
- 10:49: Correctness job: 51 steps, 0 retries, 0 rejected. **Gate item 3 fails:** the first token equals the reference on 1 of 3 prompts. `fakequant.py` reproduces the cluster's tokens.
- 10:51: The 30k session restarted.
  - The prefill ran at 39–80 chunks per 5 min.
  - GPU memory grew from 2.1 to 3.4 GB (nvidia-smi, context included).
- 11:32: **Gate item 1 fails:** w3 raised `torch.OutOfMemoryError` in attention ("Tried to allocate 234.00 MiB … 3.50 GiB allowed; 2.57 GiB is allocated by PyTorch, and 753.49 MiB is reserved by PyTorch but unallocated").
  - The coordinator did not fail the job. It waited for its 5,400 s timeout (12:21), about 50 min of idle machines.
- 12:22: `pods down do31` refused (no `runs/` folder to fetch on these machines); `pods down do31 --no-fetch` deleted all 4, and `pods ls` showed 0.

## Results

### Gate

| Item | Outcome |
|---|---|
| 1. Runs | **Fail.** The kernel, timing and correctness jobs completed with 0 retries and 0 rejected. The 30,000-token session failed on w3: out of memory at about 29,950 cached tokens (the 234 MiB request matches one float32 score matrix at that length). |
| 2. Kernel ≤ 1e-3 | **Fail** on the 3 bf16 GPUs (1.9–3.2e-3); pass on the float16 RTX 2070 (2.4–4.3e-4). |
| 3. First tokens equal | **Fail:** 1 of 3. |

### Kernel (per decode-step product, 1 row)

| GPU | Dtype | Error vs unpacked | qkv fused / unpacked | gate_up | down | head |
|---|---|---|---|---|---|---|
| RTX 4060 | bf16 | 1.9–3.1e-3 | 0.09 / 2.03 ms | 0.22 / 8.36 | 0.15 / 4.11 | 0.92 / 32.86 |
| RTX 2070 | float16 | 2.4–4.3e-4 | 0.09 / 2.24 | 0.30 / 8.58 | 0.22 / 4.14 | 0.83 / 34.04 |
| RTX 3070 | bf16 | 1.9–3.2e-3 | 0.07 / 1.42 | 0.22 / 5.99 | 0.16 / 2.93 | 0.81 / 23.84 |
| RTX 3070 | bf16 | 1.9–3.2e-3 | 0.09 / 1.29 | 0.19 / 5.43 | 0.13 / 2.65 | 0.71 / 21.65 |

On the RTX 4060, the kernel's float32 output is within 1.9–2.8e-7 (relative) of the exact product. Rounding the exact product to bf16 gives the same 1.8–3.1e-3 error as the kernel's bf16 output.

### Tokens

| Prompt | Matching tokens | Reference margin at divergence | 4-bit transformers matches the cluster |
|---|---|---|---|
| What is the capital of France? Answer in one sentence. | 16 / 16 | none | 16 / 16 |
| Write a Python function that returns the n-th Fibonacci number. | 0 | 0.875 | 16 / 16 |
| What is 17 * 23? Give only the number. | 0 | 1.625 | 3 / 16 (then an exact tie: margin 0.0) |

The replies stay coherent. Where they diverge, bf16 opens "We need to write a Python function…" and "We need compute 17*23 = 391…", and the 4-bit model opens "The user asks: …".

### Timing, 128-token prompt, 128 new tokens (3 measured sessions)

| Arm | Median inter-token | p95 | First token | Peak GPU (nvidia-smi) |
|---|---|---|---|---|
| Fused kernel | 0.951 s | 1.137 s | 10.1, 19.2, 11.9 s | 1.9–2.7 GB |
| Unpacked | 1.411 s | 1.694 s | 11.7, 18.2, 10.2 s | 1.9–2.7 GB |

The 1,024 and 8,192-token prompts were not run.

### 30,000-token session

- The prefill ran at about 4–5 s per 64-token chunk; an earlier, stopped attempt ran at about 13 s.
- GPU memory grew from 2.1 GB to a peak of 3.7–3.8 GB per machine (nvidia-smi, CUDA context included).
- w3 failed at 11:32 after about 41 minutes, with 0 retries and 0 rejected before the crash.

## What it taught

- **4 GB is enough for the weights, not for a 32k context with this allocator.**
  - At about 30k tokens, w3 had 2.57 GiB allocated and 0.75 GiB reserved but unusable for a 234 MiB request.
  - The key/value buffers grow by copying into a larger buffer every 512 positions, and each prefill chunk allocates fresh score matrices. Together they fragment torch's caching allocator.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, preallocating the cache to the session's length, or chunked attention would each likely fit. That is a new experiment, not a rescue.
- **The fused kernel works and is 20–40× faster per product, but the network sets the token time.**
  - 0.95 s per token against 1.41 s unpacked.
  - With 4 owners in 3 provinces and Nevada, each token crosses 57–88 ms links several times through the relay.
  - Compute is a few ms per owner per token.
- **Long prompts are bandwidth-bound on contributor links.**
  - Each 64-token chunk hands about 1 MB of float32 activations per hop.
  - A 30,000-token prompt moves about 0.5 GB per hop, however it is chunked, over 16–22 Mbit/s links.
- **The gate's kernel limit was wrong for bf16.** 1e-3 is below bf16's own rounding of the output (up to 3.9e-3). A float32-accumulator limit would have tested the kernel.
- **The token gate was wrong for 4-bit.** 4-bit rounding changes first tokens whose bf16 margin is under about 1.6. A 4-bit (fake-quant) reference tests the pipeline, and the bf16 one tests the quantization.
- **Bugs found:**
  - `cap_gpu_memory` needed a device index (fixed in e7bc569).
  - An owner that dies leaves the coordinator waiting until its timeout, rather than failing the job. That cost about 50 min of machines.
  - `pods down` refuses when machines have no `runs/` folder; `--no-fetch` was needed.
- **Setup is fast with range downloads:** 5 minutes for 4 machines, 3.8–4.5 GB each.

## Limits

- No real 4 GB cards: the cap stands in for one, but the GPUs are faster (2070, 3070, 4060, at 256–448 GB/s) than typical 4 GB cards (GTX 1650, T400–T1000, at 128–192 GB/s).
- Only the 128-token prompt was timed, with one job per arm (not repeated), on one pool. The 1,024 and 8,192-token timings were not run.
- The reference is transformers in bf16 on a CPU; GPU bf16 kernels may differ slightly.
- No peak-allocation counter was recorded inside torch; peaks are nvidia-smi samples, CUDA context included.
- Cost is estimated from the hourly price and Vast.ai's per-GB rate, not read from the invoice.

## Reproduce

```bash
.venv/bin/python runs/pods-data/do31/reference.py        # needs runs/models/twil-8b (17.6 GB)
.venv/bin/python runs/pods-data/do31/drive.py up          # 4 Vast.ai machines, ~$0.40/h, deleted at 3 h
.venv/bin/python runs/pods-data/do31/drive.py kernel
.venv/bin/python runs/pods-data/do31/drive.py rtt
.venv/bin/python runs/pods-data/do31/drive.py timing 128
.venv/bin/python runs/pods-data/do31/drive.py correctness
.venv/bin/python runs/pods-data/do31/drive.py long
.venv/bin/python runs/pods-data/do31/fakequant.py
python -m ygaraty.experiments.pods down do31 --no-fetch && python -m ygaraty.experiments.pods ls
```
