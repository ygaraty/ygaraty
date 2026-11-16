# The rows-per-thread kernel doubles batched throughput: eight conversations at 19.4 tokens/s against 9.8, tokens unchanged

| | |
|---|---|
| **Conclusion** | **Works:** batching 8 conversations doubles throughput, 9.8 → 19.4 tokens/s, same tokens (measured, workstation). |
| **Status** | executed |
| **ID** | WS-18 |
| **Category** | transport |
| **Where** | workstation (Radeon 680M, Chrome WebGPU) |
| **Date** | Added 2026-11-09 |
| **Path** | Qwen3.5-9B Q4_K_M, two stages (0,16,32) on the one GPU, relay route without added delay, `qwen35_session local --conversations 8 --batch`, experiment 93's chat prompts, 64 tokens each; today's kernels (B0) against `--matvec 2,4,8` (B1) |
| **Result** | **Gate met (1.98×, line 1.5×); tokens identical in every run.** Steady throughput, median of 3: B0 9.81 tokens/s, B1 19.38. A batched step of 6–7 tokens cost 363–436 ms per stage with today's kernel and 189–208 ms with the new one (one token: 56–78 ms); every conversation's latency fell from 809 to 407 ms per token. One conversation alone: 147 → 140 ms per token. The first gate script reported FAILED by also requiring 64 tokens per conversation, which conversation 3 (end-of-sequence after 24 tokens in every run, baselines included) cannot meet; the frozen criterion, equality, holds |
| **Evidence** | Measured: [evidence/browser/batched-decode-rows-per-thread-2026-11-09.json](evidence/browser/batched-decode-rows-per-thread-2026-11-09.json) (per-run throughput, per-stage compute by batch size, latencies, gate lines); raw reports in `runs/experiments/WS-18/omarchy-20261109-2341/` (not in git), with both gate-script outputs |

## Question

[Experiment 93](093-qwen35-interleaved-conversations.md) found that batching conversations gained only in
transport: a batch of 2 decode tokens cost about twice one token's compute. [Experiment 104](104-matvec-rows-per-thread.md)'s
kernel computes 8 rows for 1.3–1.7× one row's cost, bit for bit. **With it, does a batched decode step of
several conversations cost little more than one, so that eight conversations get much more total throughput
from the same GPUs?**

## Prediction

Per-stage compute of a batched step of 4–8 tokens falls from about 2–4× a single token's to about 1.3–1.7×;
steady throughput at N = 8 rises 1.5–2× (B1 against B0); every conversation's tokens unchanged.

## Gate

Frozen 2026-11-09 23:40 CET, before any run.

Arms: B0 `--conversations 8 --batch`; B1 the same with `--matvec 2,4,8`. Three repeats, order B0 B1 / B1 B0 /
B0 B1. One run each of N = 1 (T0, T1) for reference, not gated.

1. **Correctness (required):** in every B1 run, each conversation's 64 tokens equal the first B0 run's.
2. **Throughput:** the median over repeats of B1's steady throughput (report `throughput.steady_tokens_per_s`)
   at least **1.5×** B0's.
3. Report also: per-stage median compute of batched steps by batch size, N = 1 times, conversation latency.

Pass → **Works**, and the rented run (experiment 93's setup, two RTX 3060s) is proposed with its own budget.
Computed by `runs/experiments/WS-18/gate.py`.

## Protocol

`runs/experiments/WS-18/run.sh OUT PYTHON PROFILES` (the order above), then `runs/experiments/WS-18/gate.py OUT`.

## Log

- 2026-11-09 23:40 CET: added; gate frozen.
- 23:41: runs started, `runs/experiments/WS-18/omarchy-20261109-2341`.
- 23:48: eight runs finished. The gate script reported FAILED on the correctness line: it also required every
  conversation to have 64 tokens, and conversation 3 ends at end-of-sequence (token 248044) after 24 tokens in
  every run, B0 included. Tokens are identical in all runs (checked conversation by conversation; N = 1 runs
  equal conversation 0 too). The script now checks equality only, as the frozen line states; both outputs are
  in the run directory (`gate-first-script.txt`, `gate.json`). Throughput line passed in both versions.

## Results

| Arm | Steady tokens/s (3 runs) | Median | Batched step, 6–7 tokens (stage A / B) | One token (A / B) | Conversation latency |
|---|---|---|---|---|---|
| B0 today | 9.77, 9.82, 9.81 | 9.81 | 363–368 / 430–436 ms | 59.6 / 78.2 ms | 809 ms per token |
| B1 `--matvec 2,4,8` | 19.35, 19.38, 19.38 | **19.38 (1.98×)** | 189–193 / 205–208 ms | 56.4 / 74.5 ms | 407 ms per token |

With two stages and eight conversations, each stage ran batches of 6–7 tokens. N = 1: 147 ms per token today,
140 with the new kernel.

## What it taught

- **Batching now saves GPU work, not only transport.** A batched step of 6–7 tokens costs 2.8–3.4× one token
  with the new kernel, against 5.6–6.1× with today's; experiment 93's conclusion (batching gains in the relay,
  not in the GPUs) no longer holds with this kernel.
- **Throughput and latency both doubled**: every conversation's time per token halved (809 → 407 ms), because
  the stages are the bottleneck when both share one GPU.
- **Gate scripts should test exactly the frozen line.** An extra length condition turned a pass into a
  reported failure; the evidence keeps both outputs.

## Limits

- Both stages on one 680M: the stages' compute adds up; on separate GPUs (experiment 93's rented setup) the
  relay's share is larger and the gain smaller than 2×.
- One set of 8 prompts, 64 tokens.

## Reproduce

`runs/experiments/WS-18/run.sh OUT PYTHON PROFILES`, then `python runs/experiments/WS-18/gate.py OUT`.

