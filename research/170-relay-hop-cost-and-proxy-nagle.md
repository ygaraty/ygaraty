# A relayed hop costs one round trip plus about 3 ms; reports 164 and 167's extra 40 ms per hop came from the test proxy's Nagle stall

| | |
|---|---|
| **Conclusion** | Found: a relayed hop costs one round trip + 3 ms; our test proxy added 40 ms per hidden-state hop (measured, workstation) |
| **Status** | executed |
| **ID** | WS-52 (report 170) |
| **Category** | browser |
| **Where** | this machine (three native contributors on the 680M, tiny Inkling 2 bundle, relay route, a delay proxy per contributor) |
| **Priority** | high |
| **Date** | Added 2026-11-24; finished 2026-11-24 |
| **Path** | `qwen35_contributor.mjs` `?hopTrace=1` (send time on each hand-off, one `HOP` line per call: arrival, run, read-back, hand-on); `runs/experiments/WS-52/tools/delayproxy.py` in front of each contributor (round trip 2 × delay to the coordinator); `tools/hops.py` |
| **Result** | Without the network, a hop costs about 3 ms of transit (relay) or 0.3 ms (direct link); stage compute 1.5–4 ms, read-back 0.2 ms, hand-on 0.01–0.2 ms, queue 0.01 ms. With an emulated round trip r to the coordinator, transit per hop is 2.9 + 1.05–1.08 r ms (r = 6, 12, 30 ms: 9.4, 15.9, 34.4 ms): one round trip, the relay's minimum. A proxy that writes headers and body separately with Nagle on (DO-87's `proxy.py`, used on report 164's and 167's contributors) adds about 41 ms to every relayed hidden-state hop (2.9 → 44 ms; token hops, without a body, unaffected); TCP_NODELAY removes it (fixed in `proxy.py`) |
| **Evidence** | measured-local: `runs/experiments/DO-97/omarchy-20261124-local/ws52-*` (contributor logs with `HOP` lines, answers), summarized by `runs/experiments/WS-52/tools/hops.py` |

## Question

Report 167 (DO-100) measured 211–221 ms per token over the relay for a three-contributor Inkling 2 session on fast GPUs, against about 35 ms for the whole model on one A100 (report 166): about 60 ms per hop at 10–15 ms round trips. Most of a hop is therefore not network time. Which parts are: reading the hand-off back from the GPU (mapAsync), building and sending the request, the coordinator's mailbox, the next contributor's poll, uploading the hand-off, or waiting for the next stage's GPU work? The largest part decides the next change.

## Prediction

TBD. Likely candidates: Dawn's mapAsync completion in Node (event polling), the receiving contributor's poll or timer granularity, and per-token token reports on the critical path.

## Gate

A diagnostic (`Found`): the breakdown per hop, with its sum checked against the measured cycle time; no pass/fail. A fix it points to is a separate experiment.

## Protocol

1. Add per-hop timestamps (performance.now on each contributor, coordinator receive and send times in its log), off by default.
2. Tiny bundle, three contributors on this machine, relay then direct; 64-token requests. Then the real bundle where it fits (residency limits on the 680M).
3. Report the median per-hop breakdown and how it sums to the measured time per token.

Related: [167 (DO-100)](167-inkling2-fast-stages-relay-vs-turn.md), [164 (DO-97)](164-inkling2-direct-links-three-contributors.md), [161](161-resident-loop-remote.md), [157 (WS-51)](157-relay-backpressure-local.md).

## Log

- 2026-11-24: Added at the user's request, from report 167's conclusion.
- 2026-11-24: Started after the queue (user: "go back to queue and then ws52"). `?hopTrace=1` added to the contributor (off by default; answers unchanged with it on: chat, code and a 4-message chat identical to untraced runs, relay and direct).
- 2026-11-24: **Runs** (tiny bundle, 3 contributors, 48-token chat/code/multi requests; medians over 140 decode calls per stage):

  | Run | Proxy | Transit per hop A→B, B→C, C→A (ms) | Stage compute (ms) | Cycle per token (ms) | 48-token chat (s) |
  |---|---|---|---|---:|---:|
  | direct links, no proxy | none | 0.34, 0.31, 0.29 | 0.9–2.0 | 6.5 | 0.53 |
  | relay, no proxy | none | 2.3, 2.5, (token) | 3.5–3.8 | – | 1.01 |
  | relay, delay 0, Nagle on | delayproxy `--nagle` / first run | 43.6, 44.1, 3.3 | 1.5–4.0 | 101 | 4.91 |
  | relay, delay 0, TCP_NODELAY | delayproxy | 2.9, 3.0, 2.9 | 1.5–3.7 | 18.2 | 1.07 |
  | relay, r = 6 ms | delayproxy | 9.5, 9.4, 9.4 | 1.5–3.8 | 37.9 | 2.01 |
  | relay, r = 12 ms | delayproxy | 16.0, 15.9, 15.8 | 1.6–3.9 | 57.7 | 2.95 |
  | relay, r = 30 ms | delayproxy | 34.4, 34.5, 34.3 | 1.8–4.1 | 113.8 | 5.65 |
  | relay, delay 0, fixed `proxy.py` | DO-87's cache proxy with TCP_NODELAY | 2.8, 2.9, 2.9 | 1.5–3.7 | 18.1 | 1.07 |

  Read-back 0.15–0.23 ms and hand-on 0.01–0.22 ms per call in every run; queue (arrival to run) 0.01–0.05 ms.

## Conclusion

The protocol does not waste round trips: a relayed hop costs one round trip to the coordinator (one leg in, one leg out) plus about 3 ms of relay and process overhead, and the contributor's own work around a hop (read-back, encoding, hand-on, queueing) is under half a millisecond. Report 167's "about 60 ms per hop" was mostly the test harness: the weight-caching proxy on each rented contributor (DO-87's `proxy.py`) sent each response's headers and body in two writes with Nagle on, so every relayed hidden state waited about 40 ms for the receiver's delayed ACK. Hops carrying no body (tokens) were not delayed. Reports 164 and 167 therefore overstate the relay arm by about 80 ms per token (two hidden-state hops); their direct arms sent hidden states on data channels and were not affected. `proxy.py` now sets TCP_NODELAY on both sides.

## Limits

- Local, tiny model; the delay proxy adds a fixed delay with no loss or jitter. The real model's stage compute is not represented.
- The corrected relay arm of reports 164 and 167 is estimated (minus about 40 ms per hidden-state hop), not re-measured.

## Reproduce

`bash runs/experiments/WS-52/tools/session.sh runs/experiments/DO-97/omarchy-20261124-local DELAY_MS PORT TAG` (`NAGLE=1` for the stall, `PROXY=cache` for DO-87's proxy), then `python3 runs/experiments/WS-52/tools/hops.py runs/experiments/DO-97/omarchy-20261124-local/ws52-TAG`.
