# Sending the start of each link message in small pieces removes the extra round trip per decode step: 583 to 341 ms per token at 243 ms, tokens identical

| | |
|---|---|
| **Conclusion** | Works: 341 against 583 ms per token at 243 ms RTT, tokens and logits identical (measured, local TURN) |
| **Status** | executed |
| **ID** | WS-26 (experiment 153) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added and run 2026-11-11 (bench 11:28–11:40, pipeline 11:45–12:00 local) |
| **Path** | S5-nano Q4_K_M, two stages (`--splits 0,16,32`), `qwen35_session local --route webrtc --ice-policy relay`, both stage links through WS-24's local TURN server delaying each client packet by D ms; arm B unpatched `qwen35_peer.mjs` (bd3354a), arm F with the piece schedule (first 16 KiB of a body in 4 KiB pieces, the rest in 64 KiB); D = 60 order B F F B B F, then D = 0 F B |
| **Result** | One link round trip per decode step instead of two. At D = 60 (RTT 242–244 ms, `relay-relay`): B 582.8 / 583.0 / 583.3, F 340.6 / 341.1 / 340.7 ms per token, medians 583.0 against 340.7 (ratio 0.584, gate ≤ 0.75; every F below every B). At D = 0: F 92.4 against B 92.2 (within +10%). All 8 valid runs identical to WS-25's K1 in tokens and logit digests (one B0 run invalid by the frozen link rule, rerun). Now ms per token ≈ 92 + 1 × RTT. Gate: **Works**; the schedule is in `main` (4764565). Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/link-pieces-local-2026-11-11.json`; raw `runs/experiments/WS-26/omarchy-20261111-1145/main/` (`gate.json`, per-run `report.json`), bench `runs/experiments/WS-26/try-*/result.json`; scripts `runs/experiments/WS-26/{bench,go.sh,gate.py}`; code bd3354a (B), 4764565 (F, with `tests/webgpu/peer-pieces.test.mjs`) |

## Question

[WS-24](151-turn-rtt-decode-local.md) found that every decode step costs two stage-link round trips (ms per token = 91 + 2.03 × RTT): the 5,120-byte hidden state takes three one-way trips, the token one. On DO-50's 238 ms link that is about 240 ms per token of avoidable wait. Does a different piece schedule on the sender remove it, without changing a single token?

## Diagnosis (bench, before the gate)

`runs/experiments/WS-26/bench/`: one headless Chromium 155 page, two `RTCPeerConnection`s relayed through the WS-24 TURN server at D = 60 (RTT 243 ms), the decode ping-pong of the peer protocol (header, body, acks, 41/56 ms compute gaps), no model. Medians of A→B per message:

- Body 0–4,600 bytes: 122 ms (one trip, 2·D). 4,800 bytes and above (5,120; 8,192; 10,240): 364 ms (three trips). The threshold is about four packets of the 1,191-byte SCTP MTU.
- Header and body as one message: still 364 ms. A 20 ms keepalive: 139 ms (mostly, not fully). So it is not the congestion window resetting after idle (the link is never idle: acks flow both ways).
- The body in 1, 2 or 4 KiB pieces (separate `send()` calls): 122 ms for 5,120 and 10,240 bytes. The browser's SCTP sends about four packets per `send()` before waiting for an acknowledgement; more calls, one flight.
- Large bodies: 64 KiB 850 ms whole, 366 ms in 4 KiB pieces; 320 KiB 1,585 ms whole, also about 1,585 in 4 KiB pieces throughout, but 1,343 ms with the first 16 KiB in 4 KiB pieces and the rest in 64 KiB (two interleaved repeats each).

Fix (arm F): `pieces(n)` in `qwen35_peer.mjs` sends the first 16 KiB of a body in 4 KiB pieces, the rest in 64 KiB ones. The receiver already reassembles any piece sizes. Unit test `tests/webgpu/peer-pieces.test.mjs`.

## Prediction

At D = 60, ms per token drops from about 580 (B) to about 340 (F): one round trip per step instead of two. At D = 0 no change. Tokens and logits are bitwise unchanged (transport only).

## Gate (frozen 2026-11-11, before any pipeline run)

- Runs: `qwen35_session local`, S5-nano, `--splits 0,16,32 --generate 128 --logit-digests`, WS-25's prompt, `--route webrtc --ice-policy relay`, ICE = the local TURN server with TURN REST credentials; a fresh profile per run. D = 60: order B F F B B F. Then D = 0: F B.
- Valid run: state `finished`, 128 tokens and 128 digests; both stage links reported open with `relay` local and remote candidates; at D = 60 the link RTT within 25% of 243 ms. An invalid run is reported and not used.
- **Correctness (must hold):** every valid run's tokens and logit digests equal WS-25's K1 (`runs/experiments/WS-25/omarchy-20261111-0030/main/K1`), step for step.
- **Decision:** at D = 60, with M = the median over sessions of `inter_token_ms_median`: **Works** if M(F) ≤ 0.75 × M(B) and every F session is below every B session; at D = 0, F within +10% of B. Otherwise **Rejected** (or **Failed gate** on a correctness mismatch).
- Works means: ship the piece schedule; a DO-50-style relayed pair then needs about one link round trip per token.

## Protocol

- Code: arm B is a clean export of `main` at bd3354a; arm F the same plus the working tree's `qwen35_peer.mjs` and its test. Model files from `runs/models/s5-nano`, no downloads.
- TURN: `runs/experiments/WS-24/turn/turn` (unchanged), one server per run with its D.
- Under `watch.py`; one local GPU job at a time; profiles deleted after each run.

## Log

- 2026-11-11: Added at user request ("work on 2. any budget below 10 is approved"; no rental needed so far). Bench diagnosis above, then fix and unit test; gate frozen before any pipeline run.
- 2026-11-11 11:45, rehearsal (`runs/experiments/WS-26/omarchy-20261111-1145/rehearsal/`, F at D = 60): finished, 128 tokens, both links `relay-relay` at 243 ms, **341 ms per token** (WS-24 at this delay: 584), tokens and digests identical to WS-25 K1. Exports: B = bd3354a (`qwen35_peer.mjs` sha256 6b3ba5f0…), F = B plus the patched file (b3e15472…).
- 2026-11-11 11:50, main run started: `watch.py --stall 600 --max-runtime 3600`, `go.sh RUN "B60 F60 F60 B60 B60 F60 F0 B0"`.
- 2026-11-11 11:59, main run `finished`. B0-1 was **invalid** by the frozen rule: 128 tokens, all 128 messages over the link, identical to K1, but stage B's end-of-run link stats show the link `closed` (`disconnected`): stage A had already closed it at the session's end. Kept as `invalid-B0-1`, reported and not used; one replacement B0 run with the same command (12:00), valid.
- Gate by script (`gate.json`): **Works.** D = 60: B 582.8 / 583.0 / 583.3, F 340.6 / 341.1 / 340.7 ms per token (medians 583.0 and 340.7, ratio 0.584); RTT 242–244 ms, `relay-relay` on both links. D = 0: F 92.4, B 92.2 (within +10%). Correctness: all 8 valid runs identical to WS-25 K1. The prediction (about 340) held: ms per token ≈ 92 + 1 × RTT.

## Results

Gate by script (`gate.py`, recomputed 2026-11-14 from the reports with identical output):

| Run | Arm | D (ms) | Link RTT (ms) | ms per token | Identical to WS-25 K1 |
|---|---|---|---|---|---|
| B60-1 | B | 60 | 243, 244 | 582.8 | yes |
| F60-1 | F | 60 | 242, 244 | 340.6 | yes |
| F60-2 | F | 60 | 243, 243 | 341.1 | yes |
| B60-2 | B | 60 | 243, 244 | 583.0 | yes |
| B60-3 | B | 60 | 244, 244 | 583.3 | yes |
| F60-3 | F | 60 | 243, 242 | 340.7 | yes |
| F0-1 | F | 0 | 1, 1 | 92.4 | yes |
| B0-1 (rerun) | B | 0 | 1, 1 | 92.2 | yes |
| invalid-B0-1 | B | 0 | link reported closed at end | — | yes (not used) |

M(F60) / M(B60) = 340.7 / 583.0 = 0.584 ≤ 0.75, all F below all B; M(F0) = 92.4 ≤ 1.10 × 92.2; correctness holds. **Works.**

Bench (`try-*/result.json`, D = 60, medians of A→B per message, 10–20 steps each): bodies of 0–4,600 bytes 122 ms, 4,800–10,240 bytes 364 ms; header and body as one message 364 ms; a 20 ms keepalive 139 ms; the 5,120-byte body in 1 KiB pieces 122 ms, the 10,240-byte body in 1, 2 or 4 KiB pieces 122 ms; 64 KiB 850 ms whole, 366 ms in 4 KiB pieces; 320 KiB about 1,585 ms whole or all in 4 KiB pieces, 1,343–1,345 ms with the first 16 KiB in 4 KiB pieces (two interleaved repeats). B→A (the token) 122 ms throughout.

## What it taught

- Chromium's SCTP sends about four packets (of the 1,191-byte MTU) per `send()` call before waiting for an acknowledgement; a 5,120-byte hidden state is five packets, so it waited a round trip. Several `send()` calls go out in one flight. It is not the congestion window resetting after idle (WS-24's hypothesis): a keepalive barely helps.
- The fix is transport-only and bitwise neutral; on DO-50's 238 ms relayed path it should save about 240 ms per token (DO-54 is the remote check).
- **Side finding:** a stage's end-of-run link stats can report a link its peer already closed, which made one run invalid by the frozen rule.

## Limits

- One machine, integrated GPU, constant delay without loss or jitter; Chromium 155's SCTP only (Firefox and Safari not measured).
- The cause (about four packets per `send()`) is inferred from the bench, not read from the SCTP source.
- Prefill and large messages were measured in the bench only, not in a pipeline run; the bench's message pattern imitates the peer protocol but carries no model. One bench attempt (`try-113611`) produced no result.
- Three sessions per arm at D = 60 and one at D = 0.

## Reproduce

```bash
# srcB = clean export of bd3354a, srcF = the same with qwen35_peer.mjs from 4764565; WS-24's TURN built
runs/experiments/WS-26/go.sh runs/experiments/WS-26/<run-dir> "B60 F60 F60 B60 B60 F60 F0 B0"
.venv/bin/python runs/experiments/WS-26/gate.py runs/experiments/WS-26/<run-dir>    # writes gate.json
.venv/bin/python runs/experiments/WS-26/bench/bench.py runs/experiments/WS-26/<bench-dir> 60 '[{"size":5120},{"size":5120,"piece":1024}]'
node --test tests/webgpu/peer-pieces.test.mjs
```

