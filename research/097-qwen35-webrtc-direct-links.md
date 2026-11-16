# Direct WebRTC links between Qwen WebGPU stages cut time per token 61% against the coordinator relay

| | |
|---|---|
| **Conclusion** | **Works:** direct WebRTC links cut time per token 61% (211 → 83 ms), same tokens (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-40 |
| **Date** | 2026-11-08 |
| **Path** | [Experiment 95](095-qwen35-two-stages.md)'s two-stage pipeline (Qwen3.5-9B Q4_K_M, `--splits 0,16,32`, one conversation) on two Vast.ai RTX 3060 12 GB on different machines in Poland, coordinator on a DigitalOcean s-4vcpu-8gb in `ams3`. Arm R: hidden states and tokens through the coordinator (HTTP long-poll mailboxes). Arm W (`--route webrtc`): hidden states from stage A to B and tokens from B back to A over a WebRTC data channel; the coordinator only signals, records the tokens and ends the session. STUN/TURN (coturn) on the coordinator |
| **Result** | **Gate met at both lengths.** Median time per token (median of 3 runs per arm): 24-token prompt 211 → 83 ms (−61%), 8,192-token prompt 218 → 84 ms (−61%); tokens identical in all 6 runs of each length. Every link connected directly over UDP through STUN (`srflx`–`srflx`, 2–3 ms round trip); TURN was never needed. First token at 8,192 tokens 30.9 → 24.3 s. Each stage's wait per step fell from 123–159 ms to 40–45 ms, and its compute per step also fell (A 50–58 → 36–38 ms, B 78–88 → 39–40 ms). Cost about $0.75 of the $1.50 budget |
| **Evidence** | Measured: [evidence/browser/qwen35-webrtc-links-2026-11-08.json](evidence/browser/qwen35-webrtc-links-2026-11-08.json) (gate, per-run times, per-stage compute and wait, link types and round trips, tokens); raw reports in `runs/experiments/DO-40/remote-20261108-1434/main/` (not in git). Code 2fe4a98 and d04a7d4; driver `runs/experiments/DO-40/drive_remote.py`, gate `runs/experiments/DO-40/analyze.py`; tests `tests/test_qwen35_session.py` (`Signaling`) |

## Question

In [experiment 92](092-qwen35-webgpu-four-contributors.md) every hidden state and token went through the
coordinator in Amsterdam: about 270 of the 580 ms per token. With WebRTC data channels, each stage sends
straight to the next. How much of the relay time does that remove, and how often can contributors connect
directly (NAT traversal) rather than through a TURN relay?

Raised in conversation 2026-11-07.

## Prediction

Per handoff, one internet path instead of two plus the coordinator's long-poll turnaround. On contributors
in one region, relay time could fall to tens of ms per token, so 580 → about 350 ms with four stages.
Data-center containers may not allow direct UDP (Vast.ai maps ports), so some pairs may need TURN, which
brings a relay back.

## Gate

Frozen 2026-11-08, before the remote run. Same two GPUs, two stages (`--splits 0,16,32`), arms
coordinator relay (R) and WebRTC (W) interleaved R W W R R W at each prompt length (24 and 8,192
tokens), 128 tokens generated: tokens identical across every run of a length; **works** if W's median
time per token (the median of its 3 runs' medians) is at least 25% below R's at both lengths. Every W
run's links are reported (candidate types host / srflx / prflx / relay, protocol, round trip); a W run
whose link fell back to the relay or went through TURN still counts as W, and is labeled.
Computed by `runs/experiments/DO-40/analyze.py`.

## Protocol

Implemented 2026-11-08 (`--route webrtc`, job setting `route`; `--ice` for STUN/TURN servers):

- **Signaling** through the coordinator: `POST /api/signal/<id>` `{to, data}` and a long-poll
  `GET /api/signal/<id>`; SDP offers, answers and ICE candidates. One `RTCPeerConnection` per pair of
  neighbouring stages (k with k+1, the last with the first), offered by the lower stage; one ordered,
  reliable data channel (`ygaraty/webapp/static/qwen35_peer.mjs`).
- **Data:** the same float16 hidden states; a JSON header then the body in 64 KiB pieces, with
  back-pressure on the channel's buffer.
- **Tokens:** the last stage sends each token straight to the first stage and also to the coordinator
  (`direct: true`: recorded, not forwarded). The first stage stops a conversation by the coordinator's
  rule (`generate` tokens or an end token); the coordinator still ends the session with `stop`.
- **Links** open while the stages load; after `go` each stage waits up to 30 s for them. A link that does
  not open (or closes) leaves its traffic to the coordinator relay. Each stage reports its links
  (`/api/route/<id>` after go, `route` in `done`): candidate types, protocol, round trip, bytes; no
  addresses.
- Fixed prompts only (benchmarks); `--chat` keeps the relay.
- `--ice-policy relay` allows TURN only (to check the TURN path).
- **Machines:** pool `do40gpu`, 4 Vast.ai GPUs with 8 GB or more in Poland (all RTX 3060 12 GB but w2, an
  RTX A4000), and pool `do40coord`, the coordinator on DigitalOcean `s-4vcpu-8gb` in `ams3`, which also
  ran STUN/TURN (`remote.sh turn-serve`, coturn on 3478). The ICE list (with the TURN password) stayed in
  `/workspace/ice.json` and reached each W session as `--ice @/workspace/ice.json`. Stages on hosts
  `server` (A) and `w1` (B): different machines, adjacent addresses.
- **Prompts:** the 24-token default prompt and the 8,192-token prompt (`--text
  public/research/048-factorization-protein-map-webgpu.md --prompt-tokens 8192`), 128 tokens each.
- **Cost:** $0.37/h for both pools; budget $1.50, both deleted at 2 h (watchdogs `--minutes 120`).

## Log

- 2026-11-07: Added.
- 2026-11-08: Implemented and checked locally: S5-nano, two stages (`--splits 0,16,32`) in two headless
  Chromes on the workstation (Radeon 680M). One conversation, 32 tokens: tokens identical to the relay
  run; both links `host`/`host` over UDP, round trip 4–5 ms at connection; 281,600 bytes of hidden states
  sent over the channel, none through the coordinator (`hop_bytes` 0); 86 vs 89 ms per token (the relay
  is localhost here, so no gain expected). Three batched conversations (`--conversations 3 --batch`, 24
  tokens): all three identical to the relay run. Committed 2fe4a98.
- 2026-11-08: Fallback checked locally: `--ice-policy relay` with no TURN server, so no link opens; after
  the 30 s wait both stages used the coordinator relay (`hop_bytes` 281,600), tokens identical. (A first
  try with Chrome's `--force-webrtc-ip-handling-policy=disable_non_proxied_udp` did not block loopback
  UDP.) Gate frozen; committed d04a7d4.
- 2026-11-08: User's go for the remote run ("Commit, push and run remote experiment") after the proposed
  budget: **$1.50, pools deleted at 2 h**.
- ~13:15: Pools made: 4 GPUs in Poland ($0.299/h), the coordinator in `ams3` ($0.071/h); all four passed
  the WebGPU probe; all on d04a7d4. STUN/TURN up on the coordinator.
- 13:40–13:55: **First warmup abandoned** (not a measured run): w0 (192.0.2.1, listed in Poland, the
  same host that failed experiment 95's first warmup) never finished its self-test; its TCP connect to the
  coordinator took 239 ms against about 30 ms from the others. Stage B moved to w1; w0 left idle until
  teardown (`pods` deletes whole pools only).
- 13:57–14:02: Warmup W on server + w1, 8 tokens: links `srflx`/`srflx` over UDP, 2 ms round trip;
  83 ms per token.
- 14:51–15:17: The 12 measured runs, all finished with 128 tokens; `analyze.py`: gate met. p24-5R took
  about 6 minutes to start instead of about 1.5 (session setup; its timing is the same as the other R
  runs).
- 15:17: Both pools deleted (`pods down do40gpu`, `pods down do40coord`); `pods ls`: 0 machines. About
  2 hours of both pools, about $0.75 at listed prices.

## Results

| Run | Arm | Prompt tokens | ms/token (median) | p95 | First token (s) | Bytes through coordinator | Links |
|---|---|---|---|---|---|---|---|
| p24-1R | R | 24 | 211 | 283 | 1.00 | 1,236,992 | — |
| p24-2W | W | 24 | 83 | 99 | 0.97 | 0 | srflx–srflx UDP, 2 ms |
| p24-3W | W | 24 | 83 | 95 | 0.86 | 0 | srflx–srflx UDP, 2 ms |
| p24-4R | R | 24 | 211 | 280 | 0.96 | 1,236,992 | — |
| p24-5R | R | 24 | 211 | 281 | 1.15 | 1,236,992 | — |
| p24-6W | W | 24 | 83 | 93 | 1.01 | 0 | srflx–srflx UDP, 2 ms |
| p8192-1R | R | 8,192 | 217 | 285 | 30.75 | 68,149,248 | — |
| p8192-2W | W | 8,192 | 84 | 97 | 24.37 | 0 | srflx–srflx UDP, 2 ms |
| p8192-3W | W | 8,192 | 84 | 95 | 24.19 | 0 | srflx–srflx UDP, 2 ms |
| p8192-4R | R | 8,192 | 218 | 334 | 31.29 | 68,149,248 | — |
| p8192-5R | R | 8,192 | 222 | 288 | 30.88 | 68,149,248 | — |
| p8192-6W | W | 8,192 | 85 | 97 | 24.31 | 0 | srflx–srflx UDP, 2–3 ms |

**Gate** (`analyze.py`): 24 tokens: R 211.4, W 83.1 ms, cut 60.7%; 8,192 tokens: R 218.0, W 83.9 ms,
cut 61.5%; tokens identical within each length; 3 runs per arm. **Works.**

Per-stage medians per decode step (compute, wait), ms:

| Arm | Stage A | Stage B |
|---|---|---|
| R | 49–58, 150–159 | 78–88, 123–136 |
| W | 36–38, 44–45 | 39–40, 40–41 |

## What it taught

- **The relay was most of the time.** With two stages, a token's 211 ms over the relay was about 130 ms
  of compute and about 80 ms of handoffs; over WebRTC it is about 76 ms of compute and a few ms on the
  wire: each stage now waits almost exactly the other's compute time.
- **Compute fell too.** Stage B's step went from about 80 to 40 ms and A's from about 50 to 37 ms with the
  same kernels. Experiment 95 found the same thing from the other side: a GPU that idles less between steps
  keeps its clocks up. Part of R's slowness is idle GPUs, not just transfer time.
- **Direct UDP works between Vast.ai containers.** Both sides were behind NAT (`srflx`: their public
  addresses learned through STUN) and connected without TURN in every run. Residential links (home
  routers, mobile networks, symmetric NATs) are untested.
- **Prefill gains less.** First token at 8,192 tokens fell 21% (30.9 → 24.3 s): prefill is mostly compute,
  and the relay already overlapped chunks with compute.
- **Bad hosts recur.** The same Vast.ai host (192.0.2.1, 240–270 ms to Amsterdam despite being listed
  in Poland) broke the first warmup here and in experiment 95. An avoid-list that outlives one run would
  have saved about 15 minutes.

## Limits

- One pair of machines in one data center (2 ms apart). Contributors far apart, or on home links, will
  gain less in proportion: the wire time between them stays.
- Two stages and one conversation only. The code handles four stages and batched conversations (checked
  locally on S5-nano), but not measured remotely.
- TURN and the relay fallback were never exercised remotely: every link opened directly. The fallback is
  checked only locally.
- Chat sessions (`--chat`) still use the relay; restarts after a lost stage (experiment 95's recovery) are
  not wired to the WebRTC links.
- Cost is from listed prices times time, not from invoices.

## Reproduce

Workstation check (S5-nano in `runs/models/s5-nano/`):

```bash
.venv/bin/python -m ygaraty.experiments.qwen35_session local --gguf runs/models/s5-nano/S5-nano-Q4_K_M.gguf \
  --config runs/models/s5-nano/config.json --tokenizer runs/models/s5-nano/tokenizer.json --out DIR \
  --splits 0,16,32 --generate 32 --route webrtc            # and --route relay; --ice-policy relay: the fallback
```

Rented (about $0.37/h; watchdogs delete the pools):

```bash
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do40gpu --provider vast \
  --location PL --server any-8gb --workers 3 --worker-size any-8gb --minutes 120 --max-hourly 0.8 --no-collect
python -m ygaraty.experiments.pods up do40coord --provider digitalocean --location ams3 --server s-4vcpu-8gb \
  --workers 0 --minutes 120 --max-hourly 0.3 --no-collect
python -m ygaraty.experiments.pods ship do40gpu && python -m ygaraty.experiments.pods ship do40coord
# coordinator: bash /root/remote.sh qwen35-coordinator-setup && bash /root/remote.sh turn-serve
# each GPU host: bash /root/remote.sh qwen35-contributor-setup
# check each GPU host's connect time to the coordinator (curl -w '%{time_connect}'); pick two near ones
python runs/experiments/DO-40/drive_remote.py OUT/warmup --only p24-2W --generate 8 --hosts server w1
python runs/experiments/DO-40/drive_remote.py OUT/main --hosts server w1
python runs/experiments/DO-40/analyze.py OUT/main
python -m ygaraty.experiments.pods down do40gpu && python -m ygaraty.experiments.pods down do40coord && python -m ygaraty.experiments.pods ls
```
