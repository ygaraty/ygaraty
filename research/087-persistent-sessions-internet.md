# Over close internet links, persistent sessions cut TwIL-8B's time between tokens 18%, short of the 30% bar

| | |
|---|---|
| **Conclusion** | **Failed gate:** persistent sessions cut p95 inter-token latency 17.8% on four nearby hosts, below 30% (measured). |
| **Status** | executed |
| **ID** | DO-35 |
| **Date** | 2026-11-05 |
| **Path** | [Experiment 83](083-twil-int4-four-capped-gpus.md)'s setup: TwIL-8B (revision 58d49f0d) in 4-bit (layers and head), 4 owners of 10 layers on 4 Vast.ai hosts, each 8 GB GPU capped at 3,584 MiB.<br>**Hosts:** RTX 3060 Ti in Quebec (coordinator, relay, w0), RTX 3060 Ti in Ontario, RTX 3070 in Quebec, RTX 2060 SUPER in Quebec (float16).<br>**Settings:** `--dtype auto`, chunks of 64, the fused kernel, float32 wire; hosts linked only over the public internet.<br>**Arms:** A, the step-by-step path; B, persistent sessions ([experiment 84](084-persistent-phi4-sessions.md)) |
| **Result** | **Correctness passes:** identical greedy tokens across arms on experiment 83's 3 prompts and in all 16 timing sessions, 0 retries and 0 rejected in all 6 jobs.<br>**Decision gate fails:** pooled p95 inter-token latency 394.4 ms (A) against 324.2 ms (B), a 17.8% cut against a 30% bar; medians 360.9 against 308.1 ms (−14.6%).<br>**Why smaller than predicted:** this pool sat in Quebec and Ontario, so the step path ran at 361 ms rather than experiment 83's 951 ms. Most of the remaining time is the four owners' compute, which persistent sessions do not change. About $0.33 |
| **Evidence** | Measured: [evidence/transport/do35-persistent-internet-2026-11-05.json](evidence/transport/do35-persistent-internet-2026-11-05.json); raw logs in `runs/pods-data/do35/`. Code f7b3303; driver `runs/pods-data/vast_drive.py do35` |

## Question

[Experiment 84](084-persistent-phi4-sessions.md) halved the time between tokens on emulated links. [Experiment 83](083-twil-int4-four-capped-gpus.md) measured 0.95 s per token on this setup step by step, mostly network. **Over real internet links between four hosts, do persistent sessions cut the time between tokens as much, with the same tokens?**

## Prediction

- Step by step: about 0.95 s per token, as in experiment 83.
- Persistent: each hop one relay handoff (about one round trip to machine 0) plus 25–65 ms of compute per owner, about 0.4–0.5 s per token, a 45–55% cut.

## Gate (frozen 2026-11-05, before any machine was made; the user's yes)

1. **Correctness:** every job completes with 0 rejected results, and the greedy tokens are identical across arms (experiment 83's 3 chat prompts with 16 tokens, and every timing session).
2. **Decision:** arm B's p95 inter-token latency at least 30% lower than arm A's at the 128-token prompt.
3. **Reported:** jobs A B B A, each 1 warmup plus 3 measured sessions of a 128-token prompt (seed 2026) with 128 new tokens; medians and p95s, time to first token, host locations and each host's round trip to machine 0.

## Log

- 2026-11-05 16:17: Added and frozen on the user's yes ("run new do-35 and do-18"). `vast_drive.py do35 up`: 4 hosts at $0.390/h. The watchdog started as soon as they existed (the f7b3303 fix).
- 16:22: Set up in 5 min, each host downloading only its tensors.
- 16:32–16:51: 2 correctness jobs, then A B B A timing, all with 0 retries and 0 rejected.
- 16:51: Deleted; `pods ls` showed 0. Up about 34 min.

## Results

| Job | Arm | Inter-token median | p95 | First token (3 sessions) |
|---|---|---|---|---|
| timing-128-A0 | A | 368.3 ms | 397.8 ms | 2.56, 3.41, 2.79 s |
| timing-128-B1 | B | 313.5 ms | 328.5 ms | 2.44, 2.63, 2.50 s |
| timing-128-B2 | B | 281.5 ms | 321.3 ms | 2.64, 2.94, 2.56 s |
| timing-128-A3 | A | 350.8 ms | 386.3 ms | 2.55, 2.59, 2.60 s |
| **Pooled A** | | **360.9 ms** | **394.4 ms** | |
| **Pooled B** | | **308.1 ms** | **324.2 ms** | |

p95 reduction 17.8% (gate: 30%); median reduction 14.6%.

## What it taught

- **Where the hosts are sets the gain.** Experiment 83's pool spanned British Columbia, Ontario, Nevada and Quebec, and its step path ran at 951 ms per token. This pool was three hosts in Quebec and one in Ontario, and the same path ran at 361 ms.
- **What persistent sessions save:** each hop's coordinator cycle, here about 53 ms per token over 4 hops. The four owners' compute (about 25–65 ms each in experiment 83, on similar GPUs) and the relay handoffs remain.
- **The predicted 45–55% assumed experiment 83's geography,** which a Vast.ai rental does not reproduce. A fair test needs the hosts' round trips fixed or measured. This run did not measure them (`rtt` was not in the job list), so the network share is not separated from compute.
- **Same tokens over real links:** persistent sessions gave the step path's tokens in every session on four hosts, a float16 RTX 2060 SUPER among them.

## Limits

- One pool, with its own geography: the 17.8% is specific to these four hosts.
- Round trips to machine 0 were not measured.
- 128-token prompts only, 6 measured sessions per arm.
- The persistent arm has no per-owner spans (a loop task is one span), so its time is not broken down.
- Cost is estimated from the hourly price and Vast.ai's per-GB rate, not read from the invoice.

## Reproduce

```bash
.venv/bin/python runs/pods-data/vast_drive.py do35 up            # 4 Vast.ai 8 GB GPUs, ~$0.40/h, deleted at 60 min
.venv/bin/python runs/pods-data/vast_drive.py do35 correctness
.venv/bin/python runs/pods-data/vast_drive.py do35 timing
.venv/bin/python runs/pods-data/vast_drive.py do35 down && python -m ygaraty.experiments.pods ls
```
