# User sessions over direct and private WebRTC links keep the relay's tokens, but are only 11% faster between Poland and Sweden

| | |
|---|---|
| **Conclusion** | **Failed gate:** direct links keep the tokens but cut time per token only 11% (278 → 248 ms) (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-43 |
| **Date** | 2026-11-09 |
| **Path** | The user-panel path (`InferenceSession.route`, `PreparedSession`, `/api/ice`, checked signaling; `user_webrtc.md`) serving S5-nano Q4_K_M from a prepared two-stage bundle (`0,16,32`), chat through `/v1/chat/completions`. Stage A on a Vast.ai RTX 3060 in Poland, stage B on an RTX 3060 Ti in Sweden, coordinator (panel database, coordinator, coturn) on a DigitalOcean s-4vcpu-8gb in `ams3`, about 30 ms from each. Arms: R relay, W direct (`webrtc`), P direct and private (`private`: TURN only) |
| **Result** | **Correctness passed, speed failed.** All 9 sessions (R W P / P W R / W R P) gave identical texts for both gated requests; the `max_tokens` 8 request generated 8 tokens everywhere; no disagreement; no address in any route record. Median time per token (first request, median of 3 sessions): R 278 ms, W 248 ms (**−11%**, gate −25%), P 286 ms (**+3%**, gate ≤ R). In 2 of the 6 W and P sessions one link never opened (relay fallback), and 4 of 6 waited about 30 s before their first token, both caused by a negotiation race that the rehearsal fix introduced, fixed after the run. P's open links were `relay`–`relay` (62–65 ms round trip), W's `srflx`–`srflx` (45–52 ms). A local GPU screen of all three routes passed before and after the fixes. Cost about $0.25 of the $3 budget |
| **Evidence** | Measured: [evidence/browser/user-sessions-webrtc-routes-2026-11-09.json](evidence/browser/user-sessions-webrtc-routes-2026-11-09.json) (gate, per-session times, links, local screens); raw results and session logs in `runs/experiments/DO-43/` (not in git). Code e48d181, 7b45311, 9a9f9a5, 60d78e2, be4e5fb (run); negotiation fixes in this write-up's commit. Historical integration drivers are archived separately; the public evidence retains the measured route and timing results. |

## Question

[Experiment 97](097-qwen35-webrtc-direct-links.md) measured direct links on the experiment path
(`qwen35_session`). Does the user-panel path, with the security work of `user_webrtc.md` (per-slot TURN
credentials, checked signaling, peer message limits, the first stage's per-turn stop rule and token
report), keep the relay's tokens and the speed-up? And what does the private mode, which sends every
link through our TURN server, cost against the relay?

## Prediction

W close to experiment 97 (about −60% time per token against R). P pays one extra hop through the TURN
server per handoff, but no long-poll turnaround: between W and R, likely nearer W.

## Gate

Frozen 2026-11-09 11:04 CET, before any run.

**Local screen** (workstation, Radeon 680M, both stages in one GPU, two headless Chromium profiles;
no TURN server running, so P's links cannot open and must fall back to the relay):

- For each arm, one session; three chat requests: a 24-token-class prompt with `max_tokens` 32, the
  same conversation continued with `max_tokens` 6, and a new prompt with `max_tokens` 32.
- Pass if: the generated text of each request is identical across R, W and P; the `max_tokens` 6
  request generates exactly 6 tokens (or stops earlier at `<|im_end|>`) in every arm; W's links report
  open; P's links report closed and its tokens still match (relay fallback); no `disagreed` line in any
  session log; no IPv4 address in any route line of the logs.

**Remote** (two Vast.ai GPU hosts as contributors, a DigitalOcean coordinator running the coordinator,
the panel database and coturn from `scripts/remote.sh turn-serve`):

- Arms R, W, P, nine sessions in the order R W P / P W R / W R P, each with the same chat requests:
  a 24-token-class prompt with `max_tokens` 128, then the conversation continued with `max_tokens` 8.
- **Correctness:** the generated text identical across all nine sessions for each request; the second
  request generates at most 8 tokens in every session; no `disagreed` line.
- **Works** if W's median time per token (median of its 3 sessions' first-request medians, from the
  session log's `ms per token`) is at least 25% below R's, and P's at most R's.
- **Private really private:** in every P session, both ends of every link report candidate type
  `relay`; in every session, no route record or log line contains an IPv4 address.
- A W or P link that fell back to the relay still counts as its arm and is labeled.

## Protocol

`scripts/user_session_check.py` runs where the coordinator runs: for each arm it creates a session of a
two-worker S5-nano plan with that route (`create_session`), reserves both slots (`acknowledged=True`),
issues each contributor's link, starts a browser for it (`--browser-cmd`), waits for `running`, and sends
the chat requests to `/v1/chat/completions`; then it cancels the session through the administrator API.

- **Local:** `runs/experiments/DO-43/run_local.sh` starts `python -m ygaraty.coordinator.web` on
  localhost with a SQLite panel database and a bundle prepared by `ygaraty.coordinator.prepare_bundle`
  (the GGUF hard-linked), and opens each link in headless Chromium (`qwen35_session browser`, one
  profile per stage). A TURN host is configured with nothing listening.
- **Remote:** `remote.sh panel-coordinator-setup`, `turn-serve` and `panel-coordinator-serve` on the
  coordinator (https on 8443 through `https-proxy`, self-signed; browsers ignore the certificate);
  `qwen35-contributor-setup` on the GPU hosts. `remote.sh panel-check NAME` writes each contributor link
  to a 0600 file on the coordinator; `runs/experiments/DO-43/launch_remote.py` (workstation) opens each
  one with `remote.sh qwen35-browser` on the host for its stage.

## Log

- 2026-11-09 11:04 CET: added; gate frozen. Remote budget **$3**, teardown deadline **15:04 CET**
  (4 h), approved by the user. Local screen first.
- 11:06: local screen started (e48d181 + the private route, uncommitted); the first start failed at once
  (the driver could not import the package); rerun with `PYTHONPATH`.
- 11:08: **local screen passed** (see Results).
- 11:09: pool do43gpu (Vast.ai, Poland, 3 GPUs): all three were containers on one machine (one
  address), so links between them would not cross a network. Deleted at once; one GPU in Poland
  (do43a) and one in Sweden (do43b) instead, and the coordinator do43coord in `ams3`. Watchdogs
  210–219 min.
- 11:16: setup done; both GPUs about 30 ms from the coordinator.
- 11:27–11:34: rehearsal W, P (16, 4 and 8 tokens): W fine (`srflx`–`srflx`, 44 ms); **P's links did
  not open**. Cause 1: `turn-serve` denied relaying to the TURN host's own address, which is each private
  contributor's peer (both relay through it). Cause 2: the coturn package's own service, with the
  distribution's config, also ran on 3478. Fixed (9a9f9a5, 60d78e2) and applied on the coordinator by hand.
- 11:34–11:43: rehearsals 2–6 (P only): still closed. Cause 3: `turn-serve` started coturn with `-n`,
  which ignores the configuration file: no secret, realm or external address, so every allocation
  failed (STUN still worked, which is why W did). Cause 4: the first stage offered while its neighbour
  had no browser yet; the coordinator dropped that signal and the offer was never sent again (W only
  worked when B happened to be assigned first). Fixed in be4e5fb (coturn without `-n`; `reoffer()`
  after `go`); coturn restarted by hand; `qwen35_peer.mjs` (sha256 1bb6c35a…) and
  `qwen35_contributor.mjs` (6d5515c7…) copied to the coordinator.
- 11:44–11:46: rehearsal 7 (W, P) passed: W `srflx`–`srflx` 45 ms, P `relay`–`relay` 63 ms.
- 11:46–11:57: main matrix (9 sessions).
- 11:58: all three pools deleted (`pods down --no-fetch`: results and logs were already copied);
  `pods ls`: 0 machines. Gate computed: **failed** (see Results).
- 12:00–12:05: the two link failures and the 30 s waits traced to the re-offer added in be4e5fb (see What
  it taught); fixed, headless check extended to 19 checks; local screen rerun on the fixed code: passed.

## Results

### Local screen (passed, before and after the fixes)

| Arm | Tokens (requests 1–3) | Seconds | Links |
|---|---|---|---|
| R | 32, 6, 32 | 3.57, 1.26, 3.44 | — |
| W | 32, 6, 32 | 3.10, 1.17, 3.10 | `host`–`host`, `host`–`prflx` UDP, open |
| P | 32, 6, 32 | 31.56, 1.22, 3.35 | closed (no TURN server here): relay fallback |

Texts identical across the arms; no disagreement; no address in route lines. Both stages share one GPU
over loopback, so the times are not a speed measurement. P's first request includes the 30 s the stages
wait for their links after `go`. The rerun after the fixes gave the same tokens and link outcomes.

### Remote (gate failed)

Time per token of the first request (128 allowed; every session generated 52, ending at `<|im_end|>`):

| # | Arm | ms per token | First token (s) | Links |
|---|---|---|---|---|
| 1 | R | 278 | 0.95 | — |
| 2 | W | 248 | 29.1 | `srflx`–`srflx` UDP 52 / 47 ms (B's opened after 30 s) |
| 3 | P | 286 | 30.6 | `relay`–`relay` UDP 63 / 64 ms (B's opened after 30 s) |
| 4 | P | 281 | 1.20 | `relay`–`relay` UDP 65 / 64 ms |
| 5 | W | 319 | 30.9 | **did not open: relay fallback** |
| 6 | R | 278 | 1.02 | — |
| 7 | W | 241 | 1.41 | `srflx`–`srflx` UDP 45 / 45 ms |
| 8 | R | 280 | 0.96 | — |
| 9 | P | 315 | 30.6 | **did not open: relay fallback** |

| Check | Result |
|---|---|
| Request 1 and request 2 texts identical in all 9 sessions | pass |
| Request 2 at most 8 tokens | pass (8 in every session) |
| No `disagreed` line | pass |
| W at least 25% below R | **fail:** 248 vs 278 ms, −11% |
| P at most R | **fail:** 286 vs 278 ms, +3% |
| Every P link `relay`–`relay` | **fail:** session 9's link did not open |
| No IPv4 address in route records | pass |

Without the two fallback sessions, W was 241–248 ms (−12%) and P 281–286 ms (+1–3%). The 8-token
request: R 186–191 ms per token, W 147–152 ms (−20%), P 185–187 ms.

## What it taught

- **The user path is correct over all three routes.** Same tokens in every session; the per-turn
  `max_tokens` held with the first stage stopping on its own; the first stage's token report never
  disagreed; no address reached any record.
- **Direct links pay off much less here than in experiment 97.** There the two stages were 2–3 ms
  apart, so removing the coordinator's two legs and long-poll turnaround removed most of the 211 ms.
  Here the stages are 45–52 ms apart (Poland and Sweden) and the coordinator sits about 30 ms from each,
  so the direct path is not much shorter than the relayed one: about 30 ms per token saved. The gain is a
  property of the geometry, not a fixed −60%. Private links through a TURN server in the same place as the
  coordinator cost about what the relay does.
- **Four deployment bugs found in rehearsal, all in the TURN set-up and negotiation:** coturn started with
  `-n` ignored its config; the TURN host's own address was denied, which blocks every private pair; the
  coturn package started a second server on the same port; and an offer sent before the neighbour had a
  browser was dropped and never re-sent.
- **The re-offer fix itself caused the main run's failures.** It re-offered every link without an answer
  yet; when the answer to the first offer was still on its way, the offerer paired it with the new offer
  and the link never opened (sessions 5 and 9). And the stages' 30 s wait watched the link object made at
  `connect()`, so a link made again by the second offer opened unseen until the timeout (sessions 2 and 3).
  Fixed after the run: the coordinator answers each signal with `delivered`, and only an offer it dropped
  is sent again; the wait watches the current links; each link changes one step at a time (an incoming
  offer, a relink and a re-offer used to race); each stage makes its links before reading signals. The
  headless check now covers an early offer to an empty stage and a slow answer (19 checks).

## Limits

- One pair of hosts and one geometry (Poland, Sweden, coordinator in Amsterdam); experiment 97's pair
  was 2–3 ms apart. Datacenter containers, not home connections: no symmetric NAT, so W never needed TURN.
- Three sessions per arm; the fallback sessions weigh on W's and P's medians.
- The fixes made after the run were checked locally (GPU screen, headless check, unit tests), not on
  rented machines: whether every link now opens at once remotely is untested.
- S5-nano on mid-range GPUs: per-token compute was not broken out per stage (the sessions were cancelled
  without the stages' `done` reports).

## Reproduce

Local screen (S5-nano in `runs/models/s5-nano/`, a Python with the panel's web requirements):

```bash
.venv/bin/python -m ygaraty.coordinator.prepare_bundle --gguf runs/models/s5-nano/S5-nano-Q4_K_M.gguf \
  --config runs/models/s5-nano/config.json --tokenizer runs/models/s5-nano/tokenizer.json \
  --output RUN/models/s5-nano-2 --splits 0,16,32 --capacity 4096 --generate 256
runs/experiments/DO-43/run_local.sh RUN PROFILES PANEL_PYTHON
.venv/bin/python runs/experiments/DO-43/gate_local.py RUN
python scripts/webrtc_peer_check.py
```

Rented (about $0.26/h; watchdogs delete the pools):

```bash
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do43a --provider vast \
  --location PL --server any-8gb --workers 0 --minutes 215 --max-hourly 0.3 --no-collect
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do43b --provider vast \
  --location DE,CZ,NL,FR,SE,AT,BE --server any-8gb --workers 0 --minutes 212 --max-hourly 0.3 --no-collect
python -m ygaraty.experiments.pods up do43coord --provider digitalocean --location ams3 --server s-4vcpu-8gb \
  --workers 0 --minutes 210 --max-hourly 0.3 --no-collect
for p in do43a do43b do43coord; do python -m ygaraty.experiments.pods ship $p; done
# coordinator: YGARATY_QWEN_MODEL=s5-nano bash /root/remote.sh panel-coordinator-setup && bash /root/remote.sh turn-serve
#              && bash /root/remote.sh panel-coordinator-serve
# GPU hosts:   bash /root/remote.sh qwen35-contributor-setup
python -m ygaraty.experiments.pods ssh do43coord server \
  "bash /root/remote.sh panel-check main --arms R,W,P,P,W,R,W,R,P --max1 128 --max2 8 --max3 32"
python runs/experiments/DO-43/launch_remote.py main --hosts do43a:server do43b:server
# copy /workspace/panel-check-main/results.json and /workspace/serving/<session>/session.log into OUT
python runs/experiments/DO-43/gate_remote.py OUT
for p in do43a do43b do43coord; do python -m ygaraty.experiments.pods down $p --no-fetch; done
```
