# On two RTX 3060s with direct links, the rows-per-thread kernel raises eight-conversation throughput 1.86×, tokens unchanged

| | |
|---|---|
| **Conclusion** | **Works:** 8 batched conversations reach 36.2 tokens/s against 19.5, same tokens (measured, Vast.ai, 3 runs per arm). |
| **Status** | executed |
| **ID** | DO-49 |
| **Category** | transport |
| **Where** | remote |
| **Date** | Added 2026-11-10 |
| **Path** | [Experiment 97](097-qwen35-webrtc-direct-links.md)'s pipeline: Qwen3.5-9B Q4_K_M, two stages (0,16,32) on two Vast.ai RTX 3060 hosts (different machines and public IPs, Poland), direct WebRTC links (`--route webrtc`, STUN/TURN on the coordinator), coordinator DigitalOcean s-4vcpu-8gb in `ams3`; experiment 93's 8 chat prompts, 64 tokens; today's kernels (B0) against `--matvec 2,4,8` (B1); code e483d3c or later |
| **Result** | **Gate met (1.86×, line 1.3×); tokens identical in every run.** Steady throughput, median of 3: B0 19.45 tokens/s, B1 36.22. Batched steps of 3–5 tokens took 57–98 ms per stage with the new kernel against 93–206 ms for 2–6 tokens with today's; every conversation's time per token fell from 405 to 218 ms. One conversation: 106 → 104 ms per token. Links `srflx`–`srflx`, 1–2 ms. Cost about $0.16 of the $1 budget |
| **Evidence** | Measured: [evidence/browser/batched-decode-rented-2026-11-10.json](evidence/browser/batched-decode-rented-2026-11-10.json) (per-run throughput, per-stage compute by batch size, latencies, links, incidents, gate lines); raw reports and session logs in `runs/experiments/DO-49/remote-20261110-0051/` (not in git); driver `runs/experiments/DO-49/drive_remote.py`, gate `gate.py` |

## Question

[Experiment 108](108-batched-decode-rows-per-thread.md) (WS-18) doubled batched throughput on the workstation,
where both stages share one GPU. On separate GPUs with direct links, where each stage computes in parallel
with the other and transport is cheap, **does the kernel raise steady throughput for 8 batched conversations
by at least 30%, with every conversation's tokens unchanged?**

## Prediction

Batched steps of 4–8 tokens cost about 1.5–1.7× one token instead of 3–4×; steady throughput up 1.4–1.8×;
single-conversation time per token about 3% lower.

## Gate

Frozen 2026-11-10 00:50 CET, before any run.

Arms: B0 `--conversations 8 --batch --route webrtc`; B1 the same with `--matvec 2,4,8`. Three repeats, order
B0 B1 / B1 B0 / B0 B1, then one run each of N = 1 (T0, T1) for reference. All on the same two hosts.

1. **Correctness (required):** in every run, each conversation's tokens equal the first B0 run's (and T0 and
   T1 equal conversation 0's).
2. **Throughput:** the median over repeats of B1's steady throughput (`throughput.steady_tokens_per_s`) at least
   **1.3×** B0's.
3. Report also: per-stage compute of batched steps by batch size, link round trips, N = 1 times.

Budget **$1**, teardown deadline **02:50 CET** (2 h), approved by the user. Computed by
`runs/experiments/DO-49/gate.py`.

## Protocol

`remote.sh qwen35-coordinator-setup` and `turn-serve` on the coordinator; `qwen35-contributor-setup` on both GPU
hosts; a rehearsal (`--generate 8`, B1); then `runs/experiments/DO-49/drive_remote.py OUT` (one
`qwen35_session serve` per run on the coordinator, the stages' browsers on the two hosts in order).

## Log

- 2026-11-10 00:50 CET: added; gate frozen. Code e483d3c (fixed-prompt direct sessions end at the first
  stage's stop tokens; local check: 8 batched conversations over direct links, both kernels, identical tokens).
- 00:51: pools do48coord (`ams3`) and do48gpu: server RTX 3060 (machine 18578), w0 RTX 3060 (machine
  48515, another public IP in the same /24, as experiment 103's pair). A first try asking for two RTX 3060s found
  no second one and deleted its machine at once. Watchdogs end at 02:45.
- 00:53: setup done (e483d3c on all three). Rehearsal R1-1 (B1, --generate 8).
- 00:58: rehearsal passed (self-test 12 cases on both hosts, links srflx-srflx 1-2 ms, one opened after 23.3 s; 8 conversations finished). Main matrix started.
- 01:25: B0-3 failed to start: the ssh launching w0's browser timed out (60 s), the session timed out with 0 tokens (kept as `B0-3-failed-launch`). The other seven runs finished. B0-3 rerun.
- 01:27: B0-3 rerun finished; gate computed: passed. All pools deleted (`down --no-fetch`; the hosts keep no
  `runs/`); `pods ls`: 0 machines. Cost about $0.16.
- 2026-11-10 13:00 CET: ID renamed DO-48 → DO-49 before this report was first pushed: another session added an
  unrelated DO-48 (Kolibri) upstream. Scripts moved to `runs/experiments/DO-49/`.

## Results

| Arm | Steady tokens/s (3 runs) | Median | Batched step compute, stage A / B | Conversation time per token |
|---|---|---|---|---|
| B0 today | 19.29, 19.45, 20.18 | 19.45 | 2 tokens 93 / 106 ms; 5–6 tokens 174–178 / 204–206 ms | 405 ms |
| B1 `--matvec 2,4,8` | 36.46, 36.22, 35.42 | **36.22 (1.86×)** | 3–4 tokens 57–58 / 81–83 ms; 5 tokens 73 / 98 ms | 218 ms |

One conversation (N = 1): 106 ms per token today, 104 with the new kernel (stage compute 31–32 / 43 ms per
token). Every conversation's tokens equal the first B0 run's in all seven completed matrix runs and the rerun;
N = 1 runs equal conversation 0.

## What it taught

- **The workstation's 2× carries over to separate GPUs with direct links:** 1.86×, because with cheap links the
  stages' compute is the bottleneck of a batched step, and the new kernel makes extra rows nearly free (a
  5-token step costs 2.3× one token on stage A, against 5.6× before).
- **Eight conversations now get one conversation's latency doubled, not quadrupled:** 218 ms per token each,
  against 104 ms alone; with today's kernel 405 ms.
- **Single-conversation decode barely changes on the RTX 3060 (−2%)**, smaller than the 680M's −4%: the gain is
  in multi-row work (batching, speculative checks), not in plain decoding.
- **Harness:** a browser-launch ssh that times out leaves a session without a stage until its 900 s timeout;
  the driver should retry the launch.

## Limits

- One host pair at one site (1–2 ms apart); experiment 93's prompts; 64 tokens.
- Batch sizes were 2–6 tokens per stage (two stages share eight conversations), not a full 8.

## Reproduce

As the Protocol: `python runs/experiments/DO-49/drive_remote.py OUT`, then `python runs/experiments/DO-49/gate.py OUT`.

