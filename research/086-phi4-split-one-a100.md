# Splitting Phi-4 over two owners on one A100 adds 1–3 ms per token, and persistent sessions win it back

| | |
|---|---|
| **Conclusion** | **Found:** two owners add 1–3 ms per token and up to 0.73 s to first token; persistent sessions recover it (measured, A100). |
| **Status** | executed |
| **ID** | DO-18 |
| **Date** | 2026-11-05 |
| **Path** | Phi-4 (experiment 80's revision) in bf16 with a float32 head on one Vast.ai A100 PCIe 80 GB in Oklahoma. Arm 1: one owner, step by step. Arm 2: two owners (20/20 layers) as two worker processes on the same GPU, step by step. Arm 3 (extra): two owners with persistent sessions ([experiment 84](084-persistent-phi4-sessions.md)). Relay, float32 wire, chunks of 1,024 |
| **Result** | **Correctness passes:** greedy tokens identical across the three arms on experiment 80's 3 prompts and in every timing session, equal to experiment 80's float32 reference (no divergence), 0 retries and 0 rejected in all 12 jobs.<br>**Inter-token latency** (median, arms 1 / 2 / 3): 46.9 / 48.0 / 44.8 ms at 128 tokens, 46.6 / 48.8 / 45.2 at 1,024, 53.2 / 56.0 / 51.6 at 3,968.<br>**First token:** 0.055 / 0.083 / 0.073 s at 128 tokens and 0.95 / 1.68 / 1.05 s at 3,968.<br>**Weight load:** 49–50 s for one owner, 32–36 s for two. About $0.63 |
| **Evidence** | Measured: [evidence/transport/do18-phi4-split-a100-2026-11-05.json](evidence/transport/do18-phi4-split-a100-2026-11-05.json); raw logs in `runs/pods-data/do18/`. Code f7b3303; driver `runs/pods-data/vast_drive.py do18` |

## Question

Experiment 81 ran Phi-4 split over two H100s but never measured one owner on the same hardware. **With identical kernels, what does splitting Phi-4 across two owners cost per token and to first token?**

## Prediction

The split should increase single-session latency, by an unmeasured amount; it is what lets the model run on smaller devices.

## Gate (frozen 2026-11-05, before any machine was made; the user's yes)

- **Machine:** one A100 80 GB (frozen on Vultr; moved to Vast.ai before any machine was made, see Log). Phi-4 at experiment 80's revision, bf16, float32 head, relay, float32 wire, chunks of 1,024.
- **Arms:** 1, one owner (40 layers); 2, two owners (20/20) as two worker processes on the same GPU, step by step; 3 (extra, labeled), two owners with persistent sessions.
- **Correctness:**
  - greedy tokens identical across the three arms on experiment 80's 3 prompts (16 tokens) and in every timing session;
  - equal to experiment 80's float32 reference up to its first position with a top-2 margin below 0.5;
  - every job completes, with 0 rejected results;
  - no logits are exported, so the original plan's logit tolerance is replaced by these token checks.
- **Diagnostic** (no threshold): prompts of 128, 1,024 and 3,968 tokens (seed 2026), 128 new tokens, 1 warmup plus 5 measured, one job per arm and length (orders 1 2 3, 3 2 1, 2 1 3). Reported: first-token and inter-token differences with their spread, peak memory, weight-load time.

## Log

- 2026-11-04: Added; renumbered from local DO-16 to DO-18.
- 2026-11-05: Gate frozen for one Vultr A100; Vultr refused it at the account's monthly fee limit, and nothing was created.
- 16:17: Deviation, decided before any machine: a Vast.ai A100 80 GB instead, with arms, checks and measurements as frozen (the user: "run new do-35 and do-18"). Budget about $1, deleted at 60 min.
- 16:19–16:22: The first offer was gone when rented ("no_such_ask"), and nothing was created. A stale empty Vultr record (`runs/pods/do18.json`) also had to be moved aside first. Then an A100 PCIe 80 GB in Oklahoma at $0.933/h.
- 16:31: Set up in 9 min; Phi-4's 29.3 GB downloaded in 368 s.
- 16:32–16:48: 3 correctness jobs, then 9 timing jobs, all with 0 retries and 0 rejected.
- 16:48: Deleted; `pods ls` showed 0. Up about 28 min.

## Results

| Prompt | Arm | Inter-token median | p95 | First token (median of 5) | Weight load (warm-up session) | Peak GPU |
|---|---|---|---|---|---|---|
| 128 | 1 | 46.9 ms | 48.3 ms | 0.055 s | 50.2 s | 28.1 GB |
| 128 | 2 | 48.0 ms | 51.1 ms | 0.083 s | 31.7 s | 28.6 GB |
| 128 | 3 | 44.8 ms | 45.8 ms | 0.073 s | 33.2 s | 28.6 GB |
| 1,024 | 1 | 46.6 ms | 48.0 ms | 0.191 s | 48.9 s | 28.4 GB |
| 1,024 | 2 | 48.8 ms | 50.6 ms | 0.412 s | 32.1 s | 29.3 GB |
| 1,024 | 3 | 45.2 ms | 47.1 ms | 0.316 s | 35.5 s | 29.3 GB |
| 3,968 | 1 | 53.2 ms | 54.4 ms | 0.945 s | 49.4 s | 29.9 GB |
| 3,968 | 2 | 56.0 ms | 57.4 ms | 1.677 s | 33.7 s | 32.0 GB |
| 3,968 | 3 | 51.6 ms | 53.0 ms | 1.052 s | 33.6 s | 32.0 GB |

The split's cost, arm 2 minus arm 1: +1.1, +2.3 and +2.8 ms per token, and +0.03, +0.22 and +0.73 s to first token. Arm 3 minus arm 1: −2.2, −1.4 and −1.6 ms per token.

## What it taught

- **A split stage costs little per token on one GPU:** 1–3 ms, the extra hop through the relay and a second claim cycle. Persistent sessions remove the claim cycles of both owners, so two owners with them beat one owner on the step path by 1.4–2.2 ms.
- **The split's real cost is time to first token at long prompts:** +0.73 s at 3,968 tokens. Each 1,024-token prefill chunk hands 21 MB of float32 activations between owners. Persistent sessions cut that to +0.11 s, since the prefill chunks also go owner to owner without the coordinator.
- **Two owners load the weights faster:** 32–36 s against 49–50 s, since they read their halves in parallel.
- **At 47 ms per token, one A100 reads Phi-4's 29 GB at about 620 GB/s**, 32% of its 1.94 TB/s. Most of a token is still kernels and Python, not the split.

## Limits

- One GPU, one host, one job per arm and length: medians are over 5 sessions of 127 gaps, and no repeat across jobs.
- Two owners on one GPU share its bandwidth. Two GPUs would add a network link; experiment 81 measured that case.
- Peaks are nvidia-smi samples, both processes and contexts included.
- Cost is estimated from the hourly price and Vast.ai's per-GB rate, not read from the invoice.

## Reproduce

```bash
.venv/bin/python runs/pods-data/vast_drive.py do18 up            # 1 Vast.ai A100 80 GB, ~$0.93/h, deleted at 60 min
.venv/bin/python runs/pods-data/vast_drive.py do18 correctness
.venv/bin/python runs/pods-data/vast_drive.py do18 timing
.venv/bin/python runs/pods-data/vast_drive.py do18 down && python -m ygaraty.experiments.pods ls
```
