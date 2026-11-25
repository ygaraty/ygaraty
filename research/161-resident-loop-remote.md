# The resident token loop halves a lone chat's time per token on a rented A100 and keeps 8-chat throughput, with the same text

| | |
|---|---|
| **Conclusion** | Works: resident loop 33.2 vs 73.5 ms per token for one chat, 8-chat throughput 27.2 vs 24.4 tok/s, texts identical (measured, rented coordinator and A100) |
| **Status** | executed |
| **ID** | DO-87 (report 161) |
| **Category** | browser |
| **Where** | remote (local screen first) |
| **Priority** | high |
| **Date** | Added 2026-11-14; local screen 2026-11-14; remote 2026-11-24 |
| **Path** | One-stage production chat (Qwen3-Coder-30B-A3B, one contributor, DO-61 local loop, DO-85 batched reports); arms interleaved slots (`resident=False`, as deployed) vs the resident loop (experiment 129) while exactly one chat is active, back to slots when a second arrives |
| **Result** | Remote (frozen 2026-11-24 gate, same run as report 160): one chat, median of 3 blocks, **D (resident) 33.2 ms per token** (34.4, 33.2, 33.0) against **O 73.5** (82.6, 59.0, 73.5): ratio 0.45 against ≤ 0.85. 8 chats: D 27.2 tok/s (27.6, 27.2, 26.9) against O 24.4 (24.2, 24.4, 25.1): 1.12 against ≥ 0.9. 0 errors; every level's and prompt's text identical in all six D and O blocks. Descriptive: O's one chat 73.5 ms against DO-61's 133. Local screen (2026-11-14): 53.8 vs 58.3 |
| **Evidence** | measured-remote: `runs/experiments/DO-87/remote-20261124/` (`gate.json` by `tools/gate_remote.py`; blocks, coordinator and contributor logs); measured-local: `yga_old/runs/experiments/DO-87/omarchy-20261114-1250/` |

## Question

The resident loop keeps the token on the GPU: Qwen3.5-9B on an A100 went 33.4 → 22.2 ms per token (experiment
129), and for MoE the step is GPU-bound (experiment 146). Production sets `resident=False` so eight chat slots can
share the contributor. Can a one-stage session use the resident loop while one chat is active, switch cleanly
when another chat arrives, and how much does one chat gain on Qwen3-Coder?

Needs DO-85 deployed first.

## Prediction

One active chat: about 1.3–1.5× fewer ms per token, tokens identical; several chats: unchanged.

## Gate

### Local screen (frozen 2026-11-14, before the main run)

S5-nano Q4_K_M, one stage, DO-85's harness and prompts (`qwen35_session local --chat --splits 0,32 --capacity 4096
--generate 64 --record-turn-ids --loop local --relay-delay 240 --send-delay 240`, batched reports), with
production's slots: `YGARATY_CHAT_SLOTS=8` (fixture `conversations=8, batch=true`; the chat is conversation 0, the
other slots idle). Headless Chromium on the Radeon 680M, three chat turns per run. Arms:

- **A**: `YGARATY_RESIDENT=off` (as deployed: every step through the batched loop);
- **R**: resident burst while the chat is the only one decoding (DO-87 code).

Order A R R A A R. Metric as DO-85's: per run the median over its turns of client-side ms per token; per arm the
median over runs.

- **Correctness:** every run's turn ids and texts identical to A-1's; resident steps present in every R run and
  absent in every A run (the setting took effect).
- **Works** if R ≤ 0.95 × A; **Inconclusive locally** if 0.95–1.05 × A (the 680M is GPU-bound: experiment 129 saw
  only 96.2 → 90.0 ms there, 33.4 → 22.2 on an A100, so the A100 decides); **Rejected** above 1.05 × A. A
  correctness break: **Failed gate**.

Not covered locally: a second chat arriving mid-burst (the harness serves one chat). The remote arm must check
it (`chat_load` 2/4/8 texts identical across arms).

### Remote (frozen 2026-11-24, before renting; replaces the 2026-11-14 production plan below)

Setup (user's choice 2026-11-24: a rented coordinator instead of production; budget $4, every machine deleted after 150 minutes): coordinator on a Hetzner ccx33 in Nuremberg running `ygaraty serve` from this working tree (one-stage Qwen3-Coder-30B-A3B bundle prepared there from the pinned unsloth Q4_K_M GGUF, sha256 fadc3e5f…, relay route, 8 chat slots, capacity 4096); contributor on one Vast.ai A100 in the US running `native_contributor.mjs` (Node + Dawn, the same page module as Chromium) through `runs/experiments/DO-87/tools/proxy.py`, which caches only weight pieces so a fresh join loads locally and passes every API call through on a persistent connection. The load client `runs/experiments/DO-87/tools/load.py` runs on the coordinator machine (greedy, 256 tokens, fixed prompts; one chat's ms per token = (last text chunk − first) / (completion tokens − 1)). Arms are contributor-link parameters, each block a fresh join: **S** `?reports=serial&resident=off`, **O** `?resident=off` (DO-85's batched reports), **D** default (DO-87's resident burst too). Blocks S O D D O S S O D; S runs 1 chat, O and D run 1, 2, 4 and 8 chats. The contributor prints its decode mode, checked per block.

- **Correctness:** 0 errors; for every level and prompt, the same text in all six D and O blocks (a second chat arriving mid-burst is covered: a level's chats start together).
- **Works** if D's one-chat median ≤ 0.85 × O's and D's 8-chat steady tok/s median ≥ 0.9 × O's; otherwise **Rejected**; a correctness break **Failed gate**.
- Descriptive: O's one-chat ms per token against DO-61's 133 ms.

Superseded plan, kept for the record:


Production at 98cc80b or later (DO-85 and DO-87 deployed), the Qwen3-Coder one-contributor relay session, one
rented Vast.ai A100 80 GB. Blocks D O O D D O, each a fresh warm join: **D** default page (resident burst), **O**
`?resident=off` (batched loop only, DO-85 reports). In every block `chat_load` at 1, 2, 4 and 8 chats, 256 tokens,
its fixed prompts, from the workstation. One chat's ms per token is per token, not per text chunk: (last chunk −
first chunk) / (completion tokens − 1). Driver `runs/experiments/DO-87/drive_remote.py`, gate
`runs/experiments/DO-87/gate_remote.py`.

- **Correctness:** 0 errors; for every level and prompt, the same text in all six blocks (this covers a second chat
  arriving mid-burst: `chat_load` starts the chats of a level together).
- **Works** if D's one-chat median ≤ 0.85 × O's and D's 8-chat steady tok/s median ≥ 0.9 × O's; otherwise
  **Rejected**; a correctness break **Failed gate**.
- Descriptive: O's one-chat ms per token against DO-61's 133 ms (DO-85's remote effect, without its serial arm).

## Protocol

Code: page `resident` allowed with batched slots (`conversations === 1 || f.batch`); a token runs a burst when it is
the last message in hand and no batch waits (`solo`), else the batched loop; burst reports through DO-85's
`reportDirect` and with the slot's own chat metadata (`slotMeta`); production fixture `resident` on unless
`YGARATY_RESIDENT=off`; harness `YGARATY_CHAT_SLOTS`. Driver `runs/experiments/DO-87/{drive.py,go.sh,gate.py}`
(from DO-85's), rehearsal A R, then the matrix under watch.py.

## Log

- 2026-11-14: added (from the DO-85 follow-up discussion).
- 2026-11-14 12:55: started after the DO-85 deploy ("go ahead"); code on 1346f74+dirty; local gate frozen above;
  rehearsal A R under watch.py (stall 300 s, max 1200 s).
- 2026-11-14 12:50: rehearsal `omarchy-20261114-1244-rehearsal` clean: A-1 58.2, R-1 53.8 ms per token (client), texts identical, 189 resident steps in R, 189 batched steps in A (the settings took effect). Main run `runs/experiments/DO-87/omarchy-20261114-1250` A R R A A R under watch.py (stall 300 s, max 1800 s); gate `runs/experiments/DO-87/gate.py` (resident steps counted from report.json).
- 2026-11-14 13:00: main run finished (6/6 exit 0, watch.py `finished`). Gate by script: **Works**. Medians, client
  ms per token: **A 58.3, R 53.8** (ratio 0.923, limit 0.95); runs A 58.3 / 62.7 / 58.2, R 53.8 / 53.8 / 53.9;
  every run's turn ids and texts identical to A-1's; 189 resident steps in each R run, 0 in each A run. As
  experiment 129 on this GPU (−6.4%); the A100 is where the bigger gain (129: −33%) and a second chat arriving
  mid-burst get checked.
- 2026-11-14 13:20: committed 98cc80b (pushed). Remote gate frozen above; driver `runs/experiments/DO-87/drive_remote.py`,
  gate `gate_remote.py`. Production served DO-85's page but not DO-87's at the time. Paused at the user's request
  ("let it rest"): needs a deploy of 98cc80b+ and a budget (proposed cap $2.50, 100 min) before renting.
- 2026-11-24: Remote run started (user: "run in another machine"; rented coordinator instead of production; budget $4, 150-minute watchdogs). Gate frozen above. Coordinator DigitalOcean `s-8vcpu-32gb-amd` in lon1 (run `do87c`, $0.25/h; GGUF downloaded in 3.5 min, sha256 OK; bundle prepared in 3 min; `serve` needed `--memory-gb 38`, the default budget refused the 18.6 GB stage); contributor Vast.ai A100 SXM4 40 GB, US (run `do87g`, $0.563/h, driver 595.91.07). Round trip contributor → coordinator: TCP connect 251–294 ms (5 samples), so about twice production's ~120 ms. Code 12f33ea+dirty (this tree, with WS-51, DO-95 and DO-96 changes; the contributor now prints its decode mode). Blocks S O D D O S S O D under watch.py (stall 1,500 s, max 6,000 s); output `runs/experiments/DO-87/remote-20261124/`.
- 2026-11-24 23:37: The first join fetched pieces one at a time at about 6 MB/s over the 257 ms link (about 50 min for 18.5 GB). `tools/prefetch.sh` on the A100 fills the proxy's cache 16 pieces at a time (each checked against its sha256 before it is visible): about 70 MB/s. Weight loading only; no measured call goes through it.

- 2026-11-24: Attempts 1 and 2 discarded before any measurement (kept in `remote-20261124/failed/`): a driver ssh call hung during the first join (no timeout; watch.py stopped it as `stalled-idle`); then the load client sent model id "m" (the rehearsal's) and every request got 400. Fixed (every ssh bounded; the client asks `/v1/models`) and rerun from block 1. Third attempt 00:06–00:57 UTC: 9/9 blocks, 0 errors, each block's printed decode mode as its arm requires. Machines deleted by recorded ID at 01:00 (do87c about 1.6 h × $0.25, do87g about 1.6 h × $0.563: about **$1.30** of the $4).

## Limits

- One coordinator (London) and one contributor (US A100) with a 251–294 ms round trip, about twice production's; a rented `ygaraty serve` from this tree stands in for production. Native contributor (Node + Dawn), not Chromium.
- Three blocks per arm; one-chat ms per token excludes the first token.
- Weight loading went through a caching proxy (pieces only); calls went straight through on persistent connections.
- D and O differ also at 2 and 4 chats (D 17–19 against O 16–18 tok/s at 2): the burst ends as soon as a second chat arrives, as designed; the gate covers 1 and 8 chats.

Experiment 129 measured Qwen3.5 (DeltaNet), not qwen3moe; the resident path may need MoE work.

## Reproduce

Coordinator (Ubuntu, from this tree): `runs/experiments/DO-87/tools/coord.sh setup` then `coord.sh serve` (keys in /root/r/keys, src.tar.gz of this tree). Contributor (Vast.ai GPU, ubuntu:22.04, NVIDIA_DRIVER_CAPABILITIES=all): kit with `ygaraty/backends/webgpu/browser` and the DO-87 tools, `contrib.sh setup`, `contrib.sh proxy COORD:8000`, and `prefetch.sh` to fill the cache. From the workstation: `CS=... VS=... CK=... OK=... bash runs/experiments/DO-87/tools/drive.sh OUT "S O D D O S S O D"`, then `python3 runs/experiments/DO-87/tools/gate_remote.py OUT`. Code 12f33ea plus this working tree.
