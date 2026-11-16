# Each chat decode step costs two stage-link round trips, not one: 91 ms + 2.03 × RTT per token, which explains DO-50 round 2

| | |
|---|---|
| **Conclusion** | Found: ms per token = 91 + 2.03 × RTT, two link round trips per decode step (measured, local TURN) |
| **Status** | executed |
| **ID** | WS-24 (experiment 151) |
| **Category** | browser |
| **Where** | workstation |
| **Date** | Added 2026-11-10; run 2026-11-10 23:16 to 2026-11-11 00:24 local |
| **Path** | S5-nano Q4_K_M, two stages (`--splits 0,16,32`, capacity 16384, generate 256), two headless Chromium contributors and the coordinator on this machine. Route `private`: every stage link goes through a local TURN server (pion/turn) that delays each client packet by D ms each way, so a stage-to-stage round trip gains about 4·D. Arms D = 0, 30, 60 ms, order 0, 60, 30, 30, 60, 0; HTTP chat driven by `scripts/experiments/chat_cycles.py`, 3 turns of 256 tokens per session |
| **Result** | All 6 sessions valid (both links `relay-relay`, RTT 1 / 122.5–123.5 / 242–243 ms, within 25% of 4·D). Median ms per token 93 / 340–341 / 584 at RTT 1 / 123 / 242 ms (per-arm spread ≤ 1 ms); fit **k = 2.03**, intercept 90.8 ms, above the frozen 1.3: two round trips per step. First token 0.41–2.24 s (0.008 s per ms of RTT). 18/18 turns completed, no stall. A per-step probe at D = 60 puts the extra round trip on the hidden state: A→B 364 ms (three one-way trips) against B→A 122 ms (one). Gate: **Found**. At DO-50's 238 ms the link adds about 480 ms per token, so 2048 tokens need over 1000 s, beyond its 900 s bound. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/turn-rtt-decode-local-2026-11-11.json`; raw `runs/experiments/WS-24/omarchy-20261110-2316/` (`gate.json`, per-session coordinator logs and `T/events.jsonl`), probe `runs/experiments/WS-24/probe-20261111/`; scripts `runs/experiments/WS-24/gate.py`, `probe.py`, TURN server `turn/main.go` (sha256 0da7884b846c, not in git); code 64f73f3 |

## Question

In [DO-50](DO-50.md) round 2, Phi-4 on two RTX 3090 hosts (Czechia and the US) whose stage link went through TURN (238 ms) streamed tokens but never finished a 2048-token turn in 900 s: more than 0.44 s per token. The first token of a 31-token prompt took 8–26 s. How many link round trips does one decode step (and the prefill) cost? Is the slowness the link's round trip itself, or extra round trips per step that the engine could avoid?

## Prediction

One round trip per decode step (A→B for the hidden state, B→A for the token), so ms per token ≈ local compute + RTT (slope near 1). That would give about 95 + 240 ≈ 335 ms at 240 ms, not enough to explain DO-50's > 440 ms. If so, part of the remote gap is in Phi-4's compute or the TURN path, not the protocol.

## Gate (frozen 2026-11-10, before any run)

Diagnostic: the verdict is **Found**, with the slope.

- Arms: D = 0, 30, 60 ms per client leg (stage RTT ≈ 0, 120, 240 ms on top of the base), one session per arm, in the order 0, 60, 30, 30, 60, 0 (2 sessions per arm). Each session: HTTP chat, 1 cycle × 3 turns, `--words 1500 --token-cap 256 --timeout 900`.
- Valid only if: every session reaches `running` with both stage links on `relay` candidates, and for D > 0, the route log's link RTT is within 25% of 4·D plus the D = 0 RTT. An invalid session is reported, not used.
- Measure: per turn, ms per token and first-token time, from the coordinator log line (`N tokens, first after X s, Y ms per token`); the median per session; the link RTT from the route log.
- Fit ms per token = a + k · RTT over the 6 sessions, and the same for first-token time. Report k with the per-arm spread.
- **k ≤ 1.3:** one round trip per token. DO-50's slowness is the link RTT plus compute, and a round 3 needs a bound of at least 2048 × (RTT + per-token compute) or closer hosts.
- **k > 1.3:** more than one round trip per step. Found an engine cost to diagnose before round 3.
- Any turn ending with the page's 130 s no-progress error, or a `blocked` turn: a stall, reported as found.

## Protocol

- Code: a clean export of `main` at the current HEAD; the driver from the working tree (it now records the partial reply's length and the time of the last change on a `blocked` turn).
- TURN: `runs/experiments/WS-24/turn/` (Go, pion/turn, not in git): UDP on 127.0.0.1:3479, TURN REST shared secret (the coordinator's `YGARATY_TURN_SECRET`), relays on 127.0.0.1, with a per-packet delay of D ms on the client-facing socket in both directions. `YGARATY_TURN_TLS_PORT` empty; the TCP URL the coordinator offers has no listener.
- Harness: WS-22's `run.sh` and WS-21's `serve.py` with `ROUTE=private`, `YGARATY_PRIVATE_TURN_ENABLED=1`; TURN started per session with its D.
- Under `watch.py`; browser profiles in scratch, deleted after each session. One local GPU job at a time.

## Log

- 2026-11-10: Added at user request ("yes" to the proposed local reproduction of DO-50 round 2). Gate frozen above before any run.
- 2026-11-10 23:16–23:34 (rehearsal, `runs/experiments/WS-24/omarchy-20261110-2316/`, export of `main` at 64f73f3): attempt 1 set aside (a wrong run-directory argument; nothing ran). Attempt 2 set aside: pion's plain long-term auth handler rejected the coordinator's `expiry:session:slot` usernames, so the links never opened; rebuilt with `LongTermTURNRESTAuthHandler`. Attempt 3 at D = 60: both links `relay-relay`, **243 ms** (DO-50: 238 ms), opened 1.3 s after the turn started. **It reproduced DO-50 round 2:** turn 1 ended `blocked` at 900 s with the reply still growing (last change 899.9 s, 7,844 characters, first text 2.0 s); so it was slow, not stalled, at about 0.45 s per token, against 0.095 s with host links (WS-22).
- 2026-11-10 23:35, protocol fix found in rehearsal (before the main run; the gate is unchanged): `--token-cap` only sets the driver's long-output check, and the turn generates up to the bundle's `generate` (2048 in WS-22's bundle), so the frozen 256-token turns need a bundle with `--generate 256`: `models/s5-nano-256` (`prepare_bundle`, `--splits 0,16,32 --capacity 16384 --generate 256`, from `runs/models/s5-nano`, no downloads). The same flaw applies to DO-50 round 2: its 900 s bound was too short for 2048 tokens at that link's round trip.
- 2026-11-10 23:36–00:04 main run (`watch.py`, `--stall 600 --max-runtime 7200`; state `finished`), order 0, 60, 30, 30, 60, 0, bundle `s5-nano-256`. Gate by script (`gate.json`): all 6 sessions valid (`relay-relay`, RTT 1 / 242 / 122.5 / 123.5 / 243 / 1 ms, within 25% of 4·D). Median ms per token: 93, 584, 341, 340, 584, 93. Fit: **k = 2.03**, intercept 90.8 ms, so **k > 1.3: two round trips per decode step, not one**. First-token time 0.41–2.24 s (slope 0.008 s per ms). 18/18 turns completed; no stall or `blocked` turn.
- What it means for DO-50 round 2: at 238 ms, the link alone adds about 480 ms per token. 2048 tokens then need over 1000 s, beyond its 900 s bound, which matches the replies still growing at the bound. The cause of the second round trip is not identified yet. The code read so far (`qwen35_peer.mjs` `send` resolves on hand-off, not on acknowledgment; `qwen35_contributor.mjs` `handOn` / `forward`) shows no explicit wait, so it needs per-step timestamps on both stages.
- 2026-11-11 00:16–00:24, probe (user request: "run the 60ms run"), not a gate arm: one session at D = 60 from an instrumented copy of the export (`runs/experiments/WS-24/probe-20261111/`). Its `qwen35_contributor.mjs` logs each step's receive and send times, `qwen35_peer.mjs` logs each delivered link message, and `qwen35_session.py` starts Chrome with `--enable-logging=stderr`; the repository is unchanged. Still 584 ms per token, so the probe did not perturb it. Split of 762 decode steps by `probe.py` (medians, ms): A compute 41.3, **A→B link 364.1**, B queue 0.1, B compute 55.7, **B→A link 121.8**, A queue 0.1; step 583.3. The token (B→A, empty body) takes one one-way trip (2·D + 1 ms). **The hidden state (A→B, 5,120-byte body plus header) takes three one-way trips: one extra round trip per step.** Hypothesis, not tested: the SCTP data channel's congestion window restarts small after the link sits idle between steps (about 4–4.8 KB), so a 5 KB message needs a second flight after an acknowledgment. Test: shrink the hidden-state message below that size, or keep the window open, and look for A→B at one one-way trip.

## Results

Gate by script (`gate.py`, recomputed 2026-11-14 from the coordinator logs with identical output):

| Session | D (ms per leg) | Links | Link RTT (ms) | ms per token (3 turns) | First token (s) | Valid |
|---|---|---|---|---|---|---|
| m-s1-d0 | 0 | relay-relay | 1 | 93 | 0.42 | yes |
| m-s2-d60 | 60 | relay-relay | 242 | 584 | 2.24 | yes |
| m-s3-d30 | 30 | relay-relay | 122.5 | 341 | 0.84 | yes |
| m-s4-d30 | 30 | relay-relay | 123.5 | 340 | 0.96 | yes |
| m-s5-d60 | 60 | relay-relay | 243 | 584 | 2.22 | yes |
| m-s6-d0 | 0 | relay-relay | 1 | 93 | 0.41 | yes |

Fit over the 6 valid sessions: ms per token = 90.8 + **2.033** × RTT; first token = 0.3 s + 0.008 × RTT. k > 1.3: **Found**, more than one round trip per decode step. No turn ended `blocked` or with the page's no-progress error.

Probe at D = 60 (not a gate arm; instrumented copy of the same export; `probe.py` over 762 decode steps, medians in ms): A compute 41.3, **A→B link 364.1**, B queue 0.1, B compute 55.7, **B→A link 121.8**, A queue 0.1; step 583.3 (coordinator: 582–584 ms per token, so the instrumentation did not perturb it).

Rehearsal at D = 60 with a 2048-token bundle (`rh-s1-d60`, RTT 243 ms): turn 1 ended `blocked` at 900 s with the reply still growing (last change 899.9 s, 7,844 characters, first text 2.0 s): DO-50 round 2 reproduced on one machine.

## What it taught

- The token (B→A, empty body) crosses in one one-way trip; the hidden state (A→B, 5,120 bytes plus header) needs three, so one extra round trip per step. Hypothesis then: the SCTP data channel's congestion window restarts small after the link idles between steps, so a 5 KB message needs a second flight after an acknowledgment. WS-26 tested the fix (sending the start of each message in small pieces).
- DO-50 round 2's slowness was protocol, not compute: at 238 ms, two round trips add about 480 ms per token.
- **Protocol bug:** the driver's `--token-cap` only sets its long-output check; the turn length is the bundle's `generate`. Same flaw in DO-50 round 2, whose 900 s bound was too short for 2048 tokens at that RTT.
- pion's plain long-term auth handler rejects the coordinator's TURN REST usernames (`expiry:session:slot`); a local TURN needs `LongTermTURNRESTAuthHandler`.

## Limits

- One machine and an integrated GPU (S5-nano, not Phi-4): the absolute ms per token does not transfer, only the slope.
- A constant delay has no loss or jitter, unlike a transatlantic path. Both links are relayed (`private`); DO-50's pair was `srflx-relay`.
- Three RTT levels, two sessions each; the prefill's round trips were not split (first token only fitted).
- The TURN server and the probe's instrumented copy are in `runs/` (not in git); the evidence records their hashes.

## Reproduce

```bash
# clean export of 64f73f3; bundle s5-nano-256: prepare_bundle on runs/models/s5-nano --splits 0,16,32 --capacity 16384 --generate 256
# build runs/experiments/WS-24/turn (Go, pion/turn); then per session: runs/experiments/WS-24/run.sh (via go.sh, ROUTE=private YGARATY_PRIVATE_TURN_ENABLED=1)
runs/experiments/WS-24/go.sh runs/experiments/WS-24/<run-dir> "0 60 30 30 60 0" m-
.venv/bin/python runs/experiments/WS-24/gate.py runs/experiments/WS-24/<run-dir>          # writes gate.json
.venv/bin/python runs/experiments/WS-24/probe.py runs/experiments/WS-24/probe-20261111/p-s1-d60
```

