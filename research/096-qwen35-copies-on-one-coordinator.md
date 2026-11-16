# Two two-stage Qwen copies on one coordinator give 1.86× one copy's throughput at the same speed per user

| | |
|---|---|
| **Conclusion** | **Works:** two copies on one coordinator give 1.86× one copy's throughput, same per-user speed (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-41 |
| **Date** | Added 2026-11-07; run 2026-11-07 |
| **Path** | Qwen3.5-9B Q4_K_M on WebGPU contributors, each pipeline copy two stages of 16 layers ([experiment 95](095-qwen35-two-stages.md)) with batched decode across conversations ([experiment 93](093-qwen35-interleaved-conversations.md)). Copies are Qwen jobs on **one** general coordinator (`python -m ygaraty.webapp` via `remote.sh coordinator-serve`; jobs made by `qwen35_session submit` via `remote.sh qwen35-submit`), sharing its relay. Four Vast.ai RTX 3060 12 GB in Poland (copy 1 on server and w0, copy 2 on w1 and w2) and a DigitalOcean `s-4vcpu-8gb` coordinator in Amsterdam; the default chat prompts (conversation c takes message c mod 8), 128 tokens each |
| **Result** | **Part 2 passes the frozen gate** at N = 8 conversations per copy: one copy 15.6 tokens/s with its slowest conversation at 559 ms per token; two copies at once 29.1 tokens/s, slowest 557 ms (1.86× throughput, 1.00× time per token; gate 1.8× and 1.1×). **Part 1** (one copy): no N ≥ 8 stays within 1.5× of one conversation alone (218 ms): N = 8 gives 496 ms (2.28×) at 16.0 tokens/s, and throughput levels off at 16–20 tokens/s up to N = 32 (1,606 ms, 7.4×), so one two-stage copy is saturated by about 8 users. Every conversation (360 compared) equals its prompt run alone. About $0.79 of $2 |
| **Evidence** | Curated: [evidence/browser/qwen35-copies-one-coordinator-2026-11-07.json](evidence/browser/qwen35-copies-one-coordinator-2026-11-07.json) (every run and copy: state, tokens, steady tokens/s, per-conversation medians, stages; the gate's numbers). Raw: `runs/experiments/DO-41/` (not in git). Code: 7198461 (gate), deb34c8 and ae7e609 (unified coordinator), 4670983 (browser base path); driver and analysis `runs/experiments/DO-41/drive_remote.py`, `analyze.py` |

## Question

In experiment 93, eight batched conversations on one four-stage copy each got 610 ms per token, only
1.06× their time alone, so the copy was not saturated; experiment 95 then showed that two stages of
16 layers run at 246 ms per token, 57% less than four. (1) At how many conversations does one
two-stage copy's slowest conversation pass 1.5× its time alone? (2) With two copies as two jobs on one
coordinator, does total throughput double at the same per-conversation time?

Raised in conversation 2026-11-07; revised on 2026-11-07 for two-stage copies on the unified
coordinator (no routing code needed: each copy is a job with its own conversations and contributors).

## Prediction

Part 1: a 16-layer stage computes a token in 36–47 ms (experiment 95), so a batch of N tokens takes
roughly N times that (experiment 93: a batch of 2 costs about twice one); the slowest conversation
should pass 1.5× somewhere around N = 8–12. Part 2: the coordinator relays both copies' messages; at
these rates it should not be the limit, so about 2× throughput.

## Gate (frozen 2026-11-07, before any machine was made; the user's yes)

- **Correctness:** every conversation's tokens equal that conversation's prompt run alone (N = 1 runs
  with `--first-conversation k` for k = 0..7; conversation c compared with prompt c mod 8), in every run.
- **Part 1** (a finding, not pass/fail): one copy at N = 1, 8, 12, 16, 24, 32 conversations, 3 repeats
  each. Reported: the largest N whose slowest conversation's median time per token (median over
  repeats of the per-run slowest) stays within 1.5× of N = 1's median time per token; steady total
  tokens/s per N (experiment 93's definition: tokens in the window from all conversations started to
  the first one finished, over its length).
- **Part 2:** two copies at once (two jobs, each N conversations, 4 GPUs) against one copy at N, where
  N is Part 1's point (8 if that point is below 8); 3 interleaved repeats each (A B B A A B). **Works**
  if the two copies' summed steady throughput is at least 1.8× the single copy's (medians over
  repeats) and their slowest conversation's median time per token is within 1.1× of the single
  copy's.

## Protocol

- **Code:** as on main at the freeze (commit below); `--splits 0,16,32 --batch --generate 128`.
- **Local rehearsal first (workstation, 680M):** two Qwen jobs at once on one coordinator through
  `submit`, browsers on each job's join link, N = 2 per job, 8 tokens; both must finish with the
  expected tokens.
- **Pools:** `do41gpu` (4 Vast.ai RTX 3060 with NVIDIA EGL; hosts near the coordinator) and `do41coord`
  (DigitalOcean `s-4vcpu-8gb`, ams3). Contributors open their job's join link (placement fixed: copy 1
  on server and w0, copy 2 on w1 and w2).
- **Driver:** `runs/experiments/DO-41/drive_remote.py`, analysis `runs/experiments/DO-41/analyze.py`.
- **Budget:** $2, deleted at 3 h (watchdogs `--minutes 180`); expected about $0.80.

## Log

- 2026-11-07: Added.
- 2026-11-07: Revised and frozen on the user's yes ("Yes: $2, delete at 3 h"): two-stage copies as jobs on one coordinator, 4 GPUs instead of 8. Budget $2, pools deleted at 3 h.
- 2026-11-07 15:20 CET: Local rehearsal (`runs/experiments/DO-41/rehearse_local.py`, 680M): two Qwen jobs at once on one coordinator, each made by `submit`, two browsers each, 2 conversations, 8 tokens: both finished with the same tokens (conversation 0 equal to experiment 92's). Fixed before the run: `qwen35_session browser` posted its status and memory reports to the coordinator's root instead of the session's `/sessions/<job>` (4670983).
- 15:58: Pools: `do41gpu` 4 Vast.ai RTX 3060 12 GB in Poland, `do41coord` DigitalOcean `s-4vcpu-8gb` ams3; watchdogs at 18:57. `remote.sh coordinator-serve` started the general coordinator (first remote use; https 200 from every host).
- 16:08: w1 and w2 were 230–310 ms from the coordinator (TCP connect) against 30 ms for server and w0, the far "Poland" host of experiment 95; deleted by recorded ID and replaced by two RTX 3060s in Poland at 30–36 ms. All four pass the WebGPU float16 probe.
- 16:18: Warmup (both copies, 1 conversation, 8 tokens; `qwen35-submit`'s first remote use): both finished, first token 'Here' as in experiment 92; each host downloaded its stage (2.71 or 2.96 GB) once into its Chrome profile.
- 16:21–16:44: References (prompts 1–7 alone, two jobs at a time). Copy 2's w2 (192.0.2.2) had a flaky link: Chrome's TLS handshakes to the coordinator were reset (`net_error -101`; curl from w2 failed 1 of 20), so ref-3's copy 2 waited at one stage until its 900 s timeout and ref-5's copy 2 failed after 9 tokens ("Failed to fetch"). w2 was deleted by recorded ID and replaced (16:46) by an RTX 3060 on the same host network as server and w0 (20 of 20 https requests, 30 ms). Prompts 4 and 6 alone are rerun after Part 1 (on copy 2, which also loads the new w2's stage before Part 2). Copy 1 (server, w0) was not affected.
- 16:44: Part 1 started on copy 1.
- 16:44–17:37: Part 1 (18 runs, copy 1): all finished; every compared conversation equal to its prompt alone (218/218; prompts 4 and 6 had no reference yet). Medians: N = 1 218 ms per token, 4.47 tokens/s; N = 8 slowest 496 ms (2.28×), 16.0 tokens/s; N = 12 753 ms, 15.9; N = 16 975 ms, 16.2; N = 24 1,358 ms, 17.4; N = 32 1,606 ms, 19.6. No N ≥ 8 stays within 1.5×; Part 1's point is N = 1, so Part 2 runs at N = 8 (the frozen rule). The coordinator was restarted once by the driver (16:57) before its 16-job limit.
- 17:37: References for prompts 4 and 6 rerun on copy 2 (the failed ones kept as `failed-ref-3`, `failed-ref-5`), then Part 2 at N = 8.
- 17:46–18:01: Part 2 at N = 8 (A B B A A B): all finished; two copies 1.86× one copy's throughput at 1.00× its slowest time per token. The gate passes. References for prompts 4 and 6 on the new w2 finished first.
- 18:01: `pods down do41gpu` and `pods down do41coord` by recorded ID; `pods ls` showed 0 machines. About $0.79 (2 h 3 min; machines replaced after minutes counted).

## Results

**Part 1, one copy (2 GPUs), medians of 3 repeats:**

| N | Slowest conversation, ms/token | × N = 1 | Total tokens/s |
|---|---|---|---|
| 1 | 218 | 1.00 | 4.47 |
| 8 | 496 | 2.28 | 16.0 |
| 12 | 753 | 3.46 | 15.9 |
| 16 | 975 | 4.48 | 16.2 |
| 24 | 1,358 | 6.24 | 17.4 |
| 32 | 1,606 | 7.38 | 19.6 |

Part 1's point (largest N within 1.5×) is N = 1, so Part 2 ran at N = 8 (the frozen rule).

**Part 2, N = 8 per copy (A one copy, B two copies at once), in the order A B B A A B:**

| Run | Copies | Total tokens/s | Slowest ms/token |
|---|---|---|---|
| 1A | 1 | 14.05 | 578 |
| 2B | 2 | 28.66 | 557 |
| 3B | 2 | 29.10 | 569 |
| 4A | 1 | 16.12 | 499 |
| 5A | 1 | 15.62 | 559 |
| 6B | 2 | 31.59 | 539 |
| **Median** | | **A 15.62, B 29.10 (1.86×)** | **A 559, B 557 (1.00×)** |

**Correctness:** 360 of 360 compared conversations equal their prompt alone (prompts 0–7 alone on
N = 1 runs; conversation c against prompt c mod 8). Two reference runs on the first w2 did not finish
(a timeout and a failure, below) and were rerun on its replacement.

## What it taught

- **Copies add up on one coordinator.** Two jobs on the same coordinator, sharing its relay, ran as
  fast per user as one; the coordinator was not the limit at 29 tokens/s over 16 conversations.
- **One two-stage copy saturates near 8 users.** Total throughput reaches about 16 tokens/s at N = 8 and
  barely grows after (19.6 at 32), while each user's time per token grows almost in proportion to N.
  Unlike the four-stage copy of experiment 93 (8 users at 1.06× their time alone), the two-stage copy
  has little idle time to fill: its GPUs are busy, and a batch of N tokens costs close to N times one.
- **Per GPU, two stages win twice.** One two-stage copy (2 GPUs) gives the 16 tokens/s that experiment
  93's four-stage copy gave on 4 GPUs: 8 tokens/s per GPU against 4. For more users at the same speed,
  add copies; for each user's speed, keep N per copy low (about 4 users per copy stays within 1.5×,
  by interpolation; not measured).
- **The unified coordinator works remotely.** `coordinator-serve` and `qwen35-submit` ran 40 jobs on the one
  coordinator (restarted twice by the driver, between runs, before its 16-job limit), two at a time in Part 2.
- **Hosts listed in the same country can be far apart or flaky.** Two of the first four "Poland" hosts
  were 230–310 ms from the coordinator; one replacement reset TLS handshakes intermittently
  (`net_error -101`). Checking each host's https link before a run (20 requests) caught it; the driver
  still lost 15 minutes to one 900 s session timeout.
- **Bug found:** `qwen35_session browser` posted its status and memory to the coordinator's root, not
  the session's `/sessions/<job>` (fixed in 4670983, before the run).

## Limits

- One GPU model (RTX 3060) at one site; Part 2 at one N (8). Copies on other networks or with mixed GPUs
  may not add up as evenly.
- Two copies only: the coordinator's relay at 4 or more copies is not measured.
- Short chat prompts (23–30 tokens) and 128 generated tokens; all conversations start together.
- The single-copy runs (A) varied from 14.1 to 16.1 tokens/s; the two-copy runs from 28.7 to 31.6.
- w1 and w2 differ between Part 1 (unused there) and the reference reruns and Part 2 (the final w1 and
  w2); copy 1 (server, w0) is the same machines throughout.

## Reproduce

```bash
.venv/bin/python runs/experiments/DO-41/rehearse_local.py runs/experiments/DO-41/local-rehearsal   # workstation, two jobs
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do41gpu --provider vast \
  --location PL --server RTX_3060 --workers 3 --worker-size RTX_3060 --minutes 180 --max-hourly 0.45 --no-collect
python -m ygaraty.experiments.pods up do41coord --provider digitalocean --location ams3 --server s-4vcpu-8gb \
  --workers 0 --minutes 180 --max-hourly 0.3 --no-collect
python -m ygaraty.experiments.pods ship do41gpu && python -m ygaraty.experiments.pods ship do41coord
# coordinator: bash /root/remote.sh qwen35-coordinator-setup; each GPU host: bash /root/remote.sh qwen35-contributor-setup
# check each GPU host's https link to the coordinator (20 requests, connect time); replace far or flaky ones
python runs/experiments/DO-41/drive_remote.py OUT start
python runs/experiments/DO-41/drive_remote.py OUT warmup
python runs/experiments/DO-41/drive_remote.py OUT refs
python runs/experiments/DO-41/drive_remote.py OUT part1
python runs/experiments/DO-41/drive_remote.py OUT part2 --n 8
python runs/experiments/DO-41/analyze.py OUT
python -m ygaraty.experiments.pods down do41gpu && python -m ygaraty.experiments.pods down do41coord && python -m ygaraty.experiments.pods ls
```
