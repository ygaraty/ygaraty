# Batched token reports let a one-contributor chat stream at a quarter of the serial time per token across a 250 ms link

| | |
|---|---|
| **Conclusion** | Works: batched token reports 73.5 vs 292.6 ms per token for one chat at a 257 ms round trip, texts identical (measured, rented coordinator and A100) |
| **Status** | executed |
| **ID** | DO-85 (report 160) |
| **Category** | browser |
| **Where** | remote (local screen first) |
| **Priority** | high |
| **Date** | Added 2026-11-14; local screen 2026-11-14; remote 2026-11-24 |
| **Path** | Contributor page (`qwen35_contributor.mjs`): a one-stage session's self-fed tokens (DO-61 local loop) are reported to the coordinator; arms serial reports (each its own send on the serial chain, as deployed) vs batched (reports queued behind an in-flight send go out together as one `tokens` report, at most 24) |
| **Result** | Remote (frozen 2026-11-24 gate; DigitalOcean coordinator in London, Vast.ai A100 contributor in the US, Qwen3-Coder-30B-A3B one stage, resident loop off in both arms): one chat, median of 3 blocks, **O (batched) 73.5 ms per token** (82.6, 59.0, 73.5) against **S (serial) 292.6** (285.2, 326.8, 292.6): ratio 0.25 against ≤ 0.5. 0 errors; the 256-token text identical in all nine blocks. Local screen (2026-11-14): 58.4 vs 283 at an emulated 240 ms |
| **Evidence** | measured-remote: `runs/experiments/DO-87/remote-20261124/` (`gate.json` by `DO-87/tools/gate_remote.py`, shared with DO-87); measured-local: `yga_old/runs/experiments/DO-85/omarchy-20261114-0310/` |

## Question

DO-61 part 2 and DO-80 (2026-11-14, Qwen3-Coder on an A100 through production): one chat streams at 7.4 tokens per
second (133 ms per token) while the A100 needs about 25 ms per token; throughput grows linearly to 4 chats. The
page's `send` chain is serial, so each self-fed token's report waits one HTTP round trip (about 120–130 ms from
Texas) behind the previous one, and the coordinator streams one step per round trip. Do batched reports bring a
chat's ms per token to its on-machine time when each send costs a round trip?

## Prediction

Serial reports: ms per token ≈ the send round trip (when it exceeds the GPU step); batched: ≈ the on-machine
reference, tokens identical. In production about 4–5× per chat for one-contributor plans.

## Gate

### Local screen (frozen 2026-11-14, before running)

S5-nano Q4_K_M, one stage (`qwen35_session local --chat --splits 0,32 --capacity 4096 --generate 64
--record-turn-ids --loop local`, `YGARATY_RESIDENT=off` as production), headless Chromium on the Radeon 680M, one
session per run, three chat turns (`chat_cycles` prompts, cycle 0, turns 0–2, 500 words), then a graceful end.
DO-61 part 1's harness, plus `--send-delay`: every contributor send takes the emulated round trip (half before the
coordinator sees it, half before the reply). Arms:

- **S**: `YGARATY_REPORTS=serial`, `--relay-delay 240 --send-delay 240` (the deployed behavior; 240 ms as WS-24's
  243 ms link and production's 250 ms relay arm);
- **B**: batched reports (default), the same delays;
- **Z**: batched, no delays: the on-machine reference.

Order S B Z B S Z Z B S (3 runs per arm). Per run: the median over its three turns of the **client-side** ms per token
(time from the first to the last streamed text event at the chat client, divided by generated tokens − 1); per arm,
the median over its runs. (Amended 2026-11-14 after the rehearsal, before the main run: the coordinator's
`ms_per_token` measures gaps between arrivals, about 0.01 ms when a batch of tokens arrives at once, so it cannot
compare the arms.)

- **Correctness:** every run's prompt ids and generated tokens (all three turns) identical to the first Z run's.
- **Works** if B's median ≤ 1.15 × Z's + 3 ms and ≤ 0.5 × S's, with correctness. Otherwise **Rejected** (a
  correctness break: **Failed gate**). S above 200 ms confirms the cause locally (descriptive).

### Remote (frozen 2026-11-24, before renting; replaces the production plan)

Setup (user's choice 2026-11-24: a rented coordinator instead of production; budget $4, every machine deleted after 150 minutes): coordinator on a Hetzner ccx33 in Nuremberg running `ygaraty serve` from this working tree (one-stage Qwen3-Coder-30B-A3B bundle prepared there from the pinned unsloth Q4_K_M GGUF, sha256 fadc3e5f…, relay route, 8 chat slots, capacity 4096); contributor on one Vast.ai A100 in the US running `native_contributor.mjs` (Node + Dawn, the same page module as Chromium) through `runs/experiments/DO-87/tools/proxy.py`, which caches only weight pieces so a fresh join loads locally and passes every API call through on a persistent connection. The load client `runs/experiments/DO-87/tools/load.py` runs on the coordinator machine (greedy, 256 tokens, fixed prompts; one chat's ms per token = (last text chunk − first) / (completion tokens − 1)). Arms are contributor-link parameters, each block a fresh join: **S** `?reports=serial&resident=off`, **O** `?resident=off` (DO-85's batched reports), **D** default (DO-87's resident burst too). Blocks S O D D O S S O D; S runs 1 chat, O and D run 1, 2, 4 and 8 chats. The contributor prints its decode mode, checked per block.

- **Correctness:** 0 errors; the one-chat text identical in all nine blocks.
- **Works** if O's one-chat median ms per token (3 blocks) ≤ 0.5 × S's; otherwise **Rejected**; a correctness break **Failed gate**.

## Protocol

Code (page switch `?reports=serial` / fixture `report_batch`, harness `--send-delay`, `YGARATY_REPORTS=serial`),
offline tests, a rehearsal of each arm, then the local matrix under `watch.py`; driver
`runs/experiments/DO-85/{drive.py,go.sh,gate.py}` (from DO-61's).

## Log

- 2026-11-14: Added at user request ("yes, test local"), from DO-61 part 2 / DO-80's production run.
- 2026-11-14 03:05–03:10: code (uncommitted at the time): page `reportDirect` (batches of at most 24, one epoch per
  report; `reportsIdle()` awaited with `sending` before a reset and at the end), switch `?reports=serial` / fixture
  `report_batch`, harness `--send-delay` (sleep half before and half after the coordinator handles a send),
  `YGARATY_REPORTS=serial`. Offline tests pass (qwen35_session, pages, assets, notebook, chat and coordinator
  modules). Rehearsal `runs/experiments/DO-85/omarchy-20261114-0305-rehearsal/`: S 283 ms per token (client), B 58.3,
  Z 58.2, tokens identical; the coordinator's own metric reads 0.01 ms in B (batched arrivals), hence the amended
  metric.
- 2026-11-14 03:11: main run started: `runs/experiments/DO-85/go.sh runs/experiments/DO-85/omarchy-20261114-0310 "S B Z B S Z Z B S"` under watch.py (stall 300 s, max 3600 s); code 02ac1f3+dirty (DO-85 page, protocol and harness changes); gate `runs/experiments/DO-85/gate.py`.
- 2026-11-14 03:23: local screen finished (9/9 runs exit 0, watch.py `finished`). Gate by script (`runs/experiments/DO-85/omarchy-20261114-0310/gate.json`): **Works**. Medians, client ms per token: **S 283.1, B 58.4, Z 57.4** (limits: B ≤ 69.0 and ≤ 141.6); every run's turn ids and texts identical to Z-1's. With a 240 ms round trip on every send, batched reports stream at the on-machine speed (+1.7%); serial reports cost about one round trip per token (283 ≈ 240 + 43). Next: commit, deploy, then the remote arm on an A100 (`?reports=serial` vs default).
- 2026-11-24: Remote run started (user: "run in another machine"; rented coordinator instead of production; budget $4, 150-minute watchdogs). Gate frozen above. Coordinator DigitalOcean `s-8vcpu-32gb-amd` in lon1 (run `do87c`, $0.25/h; GGUF downloaded in 3.5 min, sha256 OK; bundle prepared in 3 min; `serve` needed `--memory-gb 38`, the default budget refused the 18.6 GB stage); contributor Vast.ai A100 SXM4 40 GB, US (run `do87g`, $0.563/h, driver 595.91.07). Round trip contributor → coordinator: TCP connect 251–294 ms (5 samples), so about twice production's ~120 ms. Code 12f33ea+dirty (this tree, with WS-51, DO-95 and DO-96 changes; the contributor now prints its decode mode). Blocks S O D D O S S O D under watch.py (stall 1,500 s, max 6,000 s); output `runs/experiments/DO-87/remote-20261124/`.

- 2026-11-24: Attempts 1 and 2 discarded before any measurement (kept in `remote-20261124/failed/`): a driver ssh call hung during the first join (no timeout; watch.py stopped it as `stalled-idle`); then the load client sent model id "m" (the rehearsal's) and every request got 400. Fixed (every ssh bounded; the client asks `/v1/models`) and rerun from block 1. Third attempt 00:06–00:57 UTC: 9/9 blocks, 0 errors, each block's printed decode mode as its arm requires. Machines deleted by recorded ID at 01:00 (do87c about 1.6 h × $0.25, do87g about 1.6 h × $0.563: about **$1.30** of the $4).

## Limits

- One coordinator (London) and one contributor (US A100) with a 251–294 ms round trip, about twice production's; a rented `ygaraty serve` from this tree stands in for production. Native contributor (Node + Dawn), not Chromium.
- Three blocks per arm; one-chat ms per token excludes the first token.
- Weight loading went through a caching proxy (pieces only); calls went straight through on persistent connections.
- The S arm is the behavior before DO-85; DO-85's batching is now the default, and DO-87's resident loop goes further for one chat.

## Reproduce

Coordinator (Ubuntu, from this tree): `runs/experiments/DO-87/tools/coord.sh setup` then `coord.sh serve` (keys in /root/r/keys, src.tar.gz of this tree). Contributor (Vast.ai GPU, ubuntu:22.04, NVIDIA_DRIVER_CAPABILITIES=all): kit with `ygaraty/backends/webgpu/browser` and the DO-87 tools, `contrib.sh setup`, `contrib.sh proxy COORD:8000`, and `prefetch.sh` to fill the cache. From the workstation: `CS=... VS=... CK=... OK=... bash runs/experiments/DO-87/tools/drive.sh OUT "S O D D O S S O D"`, then `python3 runs/experiments/DO-87/tools/gate_remote.py OUT`. Code 12f33ea plus this working tree.

