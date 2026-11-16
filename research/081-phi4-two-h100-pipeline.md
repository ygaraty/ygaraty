# Phi-4 split over two H100 Droplets gives the reference's tokens at 44 ms per token

| | |
|---|---|
| **Conclusion** | **Works:** two-owner Phi-4 gives experiment 80's tokens exactly; 44 ms per token (measured, 2× H100, nyc2). |
| **Status** | executed |
| **ID** | DO-14 |
| **Category** | correctness |
| **Where** | remote |
| **Date** | Added 2026-11-04, run 2026-11-04 |
| **Path** | `inference.md` stage 3. The real `microsoft/phi-4` (bf16, revision `2db69c1c`) is split over 2 DigitalOcean GPU Droplets (H100 80 GB each; planned on RTX 4000 Ada, which was out of stock) in nyc2 and its VPC. Machine A runs the coordinator, the relay and owner w0 (embedding and layers 0–19). Machine B runs owner w1 (layers 20–39, the final norm, the head and greedy selection). Torch bf16 on CUDA, with the head in float32 (experiment 80), a float32 wire, the relay and handoffs |
| **Result** | Gate passed: all 3 of experiment 80's prompts give its 16 reference tokens with no divergence at all, 0 retries, 0 rejected. Warm, per prompt length 128 / 1,024 / 3,968: median time to first token 0.07 / 0.29 / 1.12 s, median inter-token latency 44.3 / 42.3 / 43.6 ms (p95 46.3 / 43.9 / 45.0 ms), about 22.5 tokens/s while decoding. Of each 44 ms, about 37 ms is the two owners' GPU compute (17.5 + 19.4 ms) and about 7 ms the network and scheduling. Peak GPU memory 15–17 GB per owner; weight load 30.5 s. 10 minutes of 2 H100s, about $1.52 (at most $1.80) of a $15 budget |
| **Evidence** | Measured. `public/research/evidence/correctness/do14-phi4-two-h100-2026-11-04.json` (per-session times, span medians, peak memory, tokens, shard-hash check); raw run folders in `runs/pods-data/do14/` (not in git). Code commit `c5842b1` |

## Question

Experiments 79 and 80 validated our Phi-4 path on CPU. Experiment 79 used the two-worker loop on a tiny model; experiment 80 used the real checkpoint, one layer at a time. **On real GPUs and a real network, does the two-owner pipeline give the reference's tokens, and what are its warm time to first token and its inter-token latency?**

## Prediction

The tokens match experiment 80's reference, except possibly after a near-tie. Weights reads dominate compute: 14.6 GB per owner at the RTX 4000 Ada's 360 GB/s is about 40 ms per owner per token, so about 80 ms of compute per token. The VPC round trip is about 1 ms, and experiment 79's design costs about 4 round trips plus scheduling per token. Expected inter-token latency is about 90–130 ms (8–11 tokens/s). With 1,024-token chunks, the first token of a 3,968-token prompt should take under 3 s.

## Gate (frozen 2026-11-04, before any machine was made)

1. **Correctness:** use experiment 80's 3 prompts with 16 greedy tokens each, through the two-owner pipeline.
   - Pass if each prompt's tokens equal experiment 80's float32 reference up to the first position whose reference top-2 margin is below 0.5. 0.5 is above experiment 80's worst bf16 drift of 0.47.
   - A divergence at or after such a position counts as a tie and is reported. A divergence before it fails the gate.
   - Every job completes, with 0 rejected results.
2. **Measurement** (reported, not pass/fail; the plan's protocol):
   - Synthetic prompts of 128, 1,024 and 3,968 tokens (seed 2026), 128 new tokens each, prefill in chunks of 1,024.
   - 1 warmup session, which loads the weights, plus 5 measured sessions per length.
   - Reported per length: median time to first token, median and p95 inter-token latency, and tokens/s.
   - Also reported: peak GPU memory per owner, retries, and the weight-load time.
   - The one-worker baseline is labelled unavailable: the whole model (29 GB) doesn't fit in one 20 GB GPU (on H100 it would fit, but one owner is out of this gate's scope).

## Protocol

- **Machines:** `pods up do14 --provider digitalocean --location <region> --server gpu-h100x1-80gb --workers 1 --worker-size gpu-h100x1-80gb --minutes 100 --max-hourly 9` (as run; planned as `gpu-4000adax1-20gb` for 180 min).
  - DigitalOcean's NVIDIA AI/ML Ready image (`gpu-h100x1-base`).
  - Each machine downloads the whole checkpoint (the reader checks every shard's header) and records its shard hashes.
- **Setup:** `pods ship do14`, then `bash /root/remote.sh phi4-setup server` on A and `bash /root/remote.sh phi4-setup` on B, in parallel.
- **Jobs:** one `phi4-serve` job on A per arm, then `phi4-worker w0` on A and `phi4-worker w1` on B. Tokens stay in mode-600 files on the machines and are never printed.
  - Correctness: `--prompts-file` with experiment 80's prompts, `--max-new 16 --chunk 1024`.
  - Timing: `--prompt-length L --repeats 6 --max-new 128 --chunk 1024` for L of 128, 1,024 and 3,968.
- **Driver:** `runs/pods-data/do14/drive.py` (local, not in git). It fetches each run folder: the coordinator's result, worker logs and spans, and the nvidia-smi samples.
- **Teardown:** `pods down do14` when the jobs finish, then `pods ls` must show 0. The watchdog deletes the machines at 100 min regardless.

## Log

- 2026-11-04: Added. The user chose bf16 first, on 2× RTX 4000 Ada, with a $6 budget and a 3 h teardown deadline.
- 2026-11-04: Prepared at no cost:
  - the owner moves each tensor to the GPU as it is read (before this, 20 layers in float32 meant 27 GB of host RAM against the Droplet's 32 GB);
  - the head computes logits in float32;
  - the pods tool uses the GPU image for `gpu-*` sizes;
  - `remote.sh` has `phi4-setup`, `phi4-serve` and `phi4-worker`, rehearsed locally on the tiny model (2 prompts, 0 retries);
  - `phi4_run` has `--prompts-file`.
- 2026-11-04: Not started: DigitalOcean listed `gpu-4000adax1-20gb` in tor1 at 13:00, but by 13:30 it listed no region for it. The same was true of every single-GPU NVIDIA size (L40S, RTX 6000 Ada, H100, H200), in 3 checks 20 s apart. Only AMD sizes were in stock.
- 2026-11-04: Gate frozen with the user's yes. The user chose to watch stock every 10 minutes for up to 3 h and be asked before anything is made.
- 2026-11-04: Watch ended with no stock: 18 checks from 13:37 to 16:27 found no region for RTX 4000 Ada, L40S or RTX 6000 Ada. At 16:37, `gpu-h100x1-80gb` ($4.41/h) was listed in nyc2 and tor1, outside the approved scope. No machine made, $0 spent.
- 2026-11-04: The user approved 2× `gpu-h100x1-80gb` in tor1 instead: budget $15, at most 2 h. At $8.82/h the watchdog is set to 100 min ($14.70 at most). The gate is unchanged; the Prediction's latency numbers were for the RTX 4000 Ada (H100 reads weights about 9× faster, so compute is about 9 ms per token and the network and scheduling dominate).
- 2026-11-04 16:40: The first `pods up` in tor1 made the server, but tor1's last H100 went before w0 (its create returned an ID, then 404). Deleted the server with `pods down --no-fetch` after about 1 min, and made both in nyc2 at 16:41 (server 10.100.16.2, w0 10.100.16.3; watchdog at 18:20, $14.55 at most). Commit c5842b1.
- 2026-11-04 16:43: Setup on both in 2 min (torch 2.15.0+cu132, CUDA 13.2, driver 580.196.07). Both machines' shard SHA-256s equal experiment 80's.
- 2026-11-04 16:45: Correctness job: 51 steps, 0 retries, 0 rejected; all 3 prompts give the reference tokens.
- 2026-11-04 16:46–16:50: Timing jobs at 128, 1,024 and 3,968 tokens, each 1 warmup plus 5 measured sessions, 0 retries, 0 rejected.
- 2026-11-04 16:50: `pods down do14`; `pods ls` shows 0 machines.

## Results

### Correctness (gate)

| Prompt | Tokens | First reference margin below 0.5 | First divergence | Pass |
|---|---|---|---|---|
| 0 (chat, 19 tokens) | 16 | position 12 | none | yes |
| 1 (chat, 95 tokens) | 16 | position 8 | none | yes |
| 2 (synthetic, 128 tokens) | 16 | position 0 | none | yes |

All 48 tokens equal experiment 80's float32 reference, including the positions after near-ties, which the gate would have excused. Every job completed with 0 retries and 0 rejected results. The float32 head (added after experiment 80's bf16 tie) leaves no tie unresolved here.

### Timing (reported, not gated)

Warm sessions, 5 per length, 128 new tokens each, prefill in chunks of 1,024:

| Prompt tokens | Median time to first token | Inter-token median | Inter-token p95 | Tokens/s (median session) | Peak GPU memory w0 / w1 |
|---|---|---|---|---|---|
| 128 | 0.073 s | 44.3 ms | 46.3 ms | 22.4 | 14.9 / 15.2 GB |
| 1,024 | 0.290 s | 42.3 ms | 43.9 ms | 22.5 | 15.4 / 15.7 GB |
| 3,968 | 1.119 s | 43.6 ms | 45.0 ms | 19.2 | 16.7 / 17.0 GB |

- **Where a token's 44 ms go** (span medians): w0's stage compute is 17.5 ms and w1's 19.4 ms; the relay and posts take about 1–2 ms each, and the rest is claims and scheduling. Compute is about 37 ms, the network and scheduling about 7 ms.
- **Weight load:** 30.5 s per owner, which makes each job's warmup session's first token about 41 s.
- **One-worker baseline:** not measured. The gate scoped it out because the model doesn't fit a 20 GB GPU; on the H100s it would fit, which is a separate question.

## What it taught

- **The pipeline is correct on real GPUs.** Experiment 79's loop and experiment 80's checks carry over to CUDA bf16 with a real VPC unchanged: no retries and no token differences.
- **Compute, not the network, sets the speed, but not at the hardware's bandwidth.** The prediction expected weight reads to dominate. They do, but at batch 1 each owner reads 14.6 GB in 17.5–19.4 ms, about 0.8 TB/s, a quarter of the H100's 3.35 TB/s. That is about 2× the ~9 ms/token predicted for H100 compute. The likely cost is the explicit attention matrix products and the many small kernel launches of each layer (no CUDA graphs and no fused attention); this run doesn't separate them.
- **Two owners add little.** About 7 ms per token is network and scheduling: about 16% of the token time across two machines.
- **Long prompts are cheap to admit.** 3,968 tokens reach the first token in 1.1 s; the plan's goal was under 3 s.
- **Stock is the practical constraint.** The cheap GPUs were out of stock for over 3 hours, and tor1's last H100 went between the pool's two creates.

## Limits

- Run on H100s, not the RTX 4000 Ada the prediction was made for; latency on cheaper GPUs is still unmeasured.
- One pool, one region, one run per length: no repeat on a second pool.
- Batch 1, one session at a time: no throughput across concurrent conversations.
- The peak GPU memory is from 1 s nvidia-smi samples, and includes torch's cache.
- Cost is computed from the machines' lifetimes and list price, not read from the invoice.

## Reproduce

```bash
python -m ygaraty.experiments.pods up do14 --provider digitalocean --location nyc2 --server gpu-h100x1-80gb \
  --workers 1 --worker-size gpu-h100x1-80gb --minutes 100 --max-hourly 9
python -m ygaraty.experiments.pods ship do14
# in parallel: server `bash /root/remote.sh phi4-setup server`, w0 `bash /root/remote.sh phi4-setup`
.venv/bin/python runs/pods-data/do14/drive.py correctness
.venv/bin/python runs/pods-data/do14/drive.py timing 128 1024 3968
.venv/bin/python runs/pods-data/do14/analyze.py
python -m ygaraty.experiments.pods down do14 && python -m ygaraty.experiments.pods ls
```

`drive.py` and `analyze.py` live in `runs/pods-data/do14/` (not in git). The driver runs `phi4-serve` on A, copies the tokens to B through pipes, starts `phi4-worker w0` on A and `phi4-worker w1` on B, and fetches the run folders.
