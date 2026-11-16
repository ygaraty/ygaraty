# One S5-nano contributor serves both relay and direct chat on the workstation: 60 of 60 turns, 2.91 GB allocated

| | |
|---|---|
| **Conclusion** | Works: 60 of 60 one-worker chat turns complete on relay and direct chat (measured, workstation) |
| **Status** | executed |
| **ID** | WS-23 (experiment 150) |
| **Category** | browser |
| **Where** | workstation |
| **Date** | Added and run 2026-11-10 (runs 19:45–20:10 local) |
| **Path** | S5-nano Q4_K_M, one contributor owning layers [0, 32) (embedding and head included), one headless Chromium contributor and the coordinator on this machine; the real chat page driven by `scripts/experiments/chat_cycles.py`; relay-route session with HTTP chat (R), WebRTC-route session with HTTP chat (H1, H2) and direct chat (D); capacity 4096, cap 256 |
| **Result** | Both sessions ran with the single contributor; R, H1, D and H2 each 15/15 turns and every reset completed with Send unlocked, 0 runner or page errors; direct chat connected to the single contributor; all 60 turns reached the 256 cap; median first text 0.61 s, slowest turn 31.0 s. The contributor loaded 2.70 GB and allocated 2.91 GB (14.2 s cold, 5.4 s from the browser cache). Gate: **Works**. The text-parity numbers first reported (H1 = H2 on 7 of 15 turns) compared reply plus timing line, so parity was not measured here; WS-27 later showed the chat path bitwise repeatable. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/one-worker-chat-local-2026-11-10.json`; raw `runs/experiments/WS-23/omarchy-20261110-1921/` (`gate.json`, per-run `events.jsonl`, coordinator logs, `browsers/stage0/memory_summary.json`); code 377a646 |

## Question

[DO-51](DO-51.md)'s one-worker path (one contributor is both chat endpoints; 805cfa9) was checked offline only: no browser, WebRTC connection or GPU has run it. Before renting an 8 GiB GPU for DO-51: does a one-worker session load and answer on both routes, does direct chat connect to the single contributor and complete repeated turns, and do relay and direct produce the same text?

This screen does **not** decide DO-51's memory gate: the workstation's GPU (Radeon 680M) shares system memory, so it is not an 8 GiB device. Memory is recorded descriptively.

## Prediction

The session loads (2.6 GiB of weights) and every turn completes on both routes. Direct chat connects, since no stage links are needed. Texts are identical across arms if the chat decodes greedily with the same kernels.

## Gate (frozen 2026-11-10, before any run)

One bundle (`--splits 0,32 --capacity 4096 --generate 256`), plan with one worker, 8 GB. Runs in this order:

| Run | Session route | Chat | Conversations × turns |
|---|---|---|---|
| R | relay | HTTP | 3 × 5 |
| H1 | webrtc | HTTP | 3 × 5 |
| D | webrtc (same session as H1) | direct | 3 × 5 |
| H2 | webrtc (same session) | HTTP | 3 × 5 |

- **Pass (feasibility and reliability):** both sessions reach `running` with the single contributor; every attempted turn and reset `completed` with Send unlocked; no runner errors; D reaches "WebRTC chat connected" (a fallback fails the run).
- **Long output tested:** at least 10 of 15 turns per run reach 205 tokens; otherwise reported as untested.
- **Text parity (descriptive, gated on repeatability):** per-turn output hashes compared by (conversation, turn). If H1 = H2 on every turn (repeatable), then D = H1 and R = H1 are required for "parity holds"; any difference is reported as a parity failure to diagnose. If H1 ≠ H2, parity is reported as undetermined (the decode is not repeatable on this setup), not as a failure. Text hashes are not token or logit equality; DO-51 keeps the bitwise logit gate.
- Outcome: pass → **Works** locally; DO-51 goes ahead on a real 8 GiB GPU (budget to be asked). Any failure → **Found**: diagnose before renting.

## Protocol

- Code: a clean export of `main` at 377a646 (includes 805cfa9's one-worker path and the chat UI changes up to that commit); the driver from the working tree.
- Bundle from `ygaraty.coordinator.prepare_bundle` on `runs/models/s5-nano` (no downloads), `--splits 0,32 --capacity 4096 --generate 256`.
- Harness: `runs/experiments/WS-22/run.sh` with `runs/experiments/WS-21/serve.py` (now taking `WORKERS`, `ROUTE` from the environment), `WORKERS=1 SPLITS=0,32 MEMORY_GB=8`; one run.sh call per session (R; then H1, D, H2). Coordinator on `localhost:8090`, TURN host with nothing listening; driver in the scratch Playwright 1.65.0 venv with the system Chromium, `--words 500`, default 150 s bound. Chat URL only in a mode-600 file and the driver's environment.
- Runs only after WS-22 ends (one GPU, one port). Under `watch.py` (`--stall 600 --watch`, `--max-runtime 7200`). Gate computed by script from `events.jsonl`.

## Log

- 2026-11-10: Added at user request ("start to work on our s5-nano single worker distribution plan") as DO-51's local screen. Gate frozen above before running.
- 2026-11-10 19:21: clean export of 377a646 into `runs/experiments/WS-23/omarchy-20261110-1921/src`; one-worker bundle prepared (`models/s5-nano-1w`, one stage of 2.6 GiB).
- 2026-11-10 19:45–20:10: runs under `go.sh` (queued behind WS-22), `watch.py` limits as in the Protocol; both watchers `finished`.
- 2026-11-10 20:10: **GATE PASS** for feasibility and reliability (`gate.json`, by script): R 15/15 (relay route), H1 15/15, D 15/15 (direct chat connected to the single contributor), H2 15/15; every reset completed; 0 runner or page errors; all 60 turns at the 256 cap; median first visible text 0.61 s, slowest turn 31 s. The contributor loaded 2.70 GB, allocated 2.91 GB (session log), in 14.2 s cold (download from the coordinator) and 5.4 s from the browser cache. **Text parity undetermined:** H1 = H2 on only 7 of 15 turns, so the decode is not repeatable on this setup; D = H1 on 5 of 15, R = H1 on 1 of 15. Even a fresh conversation's first turn (identical prompt) differs between runs.
- Cross-check (same script on earlier runs): this is not specific to one worker. Paired HTTP and direct runs agree on 21/50 turns (WS-21, 256), 8/12 (WS-21, 1024), 7/30 and 2/12 (DO-50, Phi-4 on an RTX 3090), 2/10 (WS-22); first turns of a fresh conversation disagree in most pairs. The page text hash cannot say where the tokens diverge; that needs token or logit capture (DO-51's gate 2 asks for bitwise logits, which presupposes repeatability).
- 2026-11-11: Correction from [WS-27](154-chat-repeatability-local.md): the text-parity numbers above (H1 = H2 on 7 of 15, D = H1 on 5, R = H1 on 1, and the cross-checks) compared the reply **plus the page's timing line**; hashes matched exactly when the displayed timings did (15/15). The chat path is bitwise repeatable (WS-27), so these numbers say nothing about text parity. The driver now hashes the reply only.

## Results

Gate by script (`gate.json`, recomputed 2026-11-14 from the events by `runs/experiments/WS-23/gate23.py` with identical output):

| Run | Session route | Chat | Turns completed | Resets | At 256 cap | Max turn | Median first text |
|---|---|---|---|---|---|---|---|
| R | relay | HTTP | 15/15 | 3/3 | 15 | 25.5 s | 0.61 s |
| H1 | webrtc | HTTP | 15/15 | 3/3 | 15 | 31.0 s | 0.61 s |
| D | webrtc | direct | 15/15 | 3/3 | 15 | 24.5 s | 0.61 s |
| H2 | webrtc | HTTP | 15/15 | 3/3 | 15 | 26.3 s | 0.61 s |

0 runner and page errors; the only failed requests are the page's own `net::ERR_ABORTED` on `/api/chat` after each stream (the direct arm also posts `/api/chat`, carrying the direct lease). The driver waits for "WebRTC chat connected" before D's first turn and times out after 40 s otherwise; D completed all its turns. Long output tested (15 of 15 per run, needed 10). **GATE PASS: Works.**

Parity as hashed by the driver then (descriptive, not valid; see Log 2026-11-11): H1 = H2 on 7 of 15, D = H1 on 5, R = H1 on 1.

Memory (descriptive; the 680M shares system memory): coordinator logs 2.91 GB allocated by the contributor; Chromium peak total RSS 1.96 GB, GPU process DRM VRAM total 2.98 GB.

## What it taught

- The one-worker path (805cfa9: one contributor is both chat endpoints) works on a real browser and GPU, on both routes and with direct chat, with no stage links.
- **Measurement bug:** the driver hashed the reply together with the page's timing line, so identical replies looked different whenever timings differed. This made the decode look non-repeatable here and in the cross-checks; WS-25 and WS-27 traced it, and the driver now hashes the reply only.
- DO-51's memory question needs a real 8 GiB device: this GPU's numbers include shared system memory.

## Limits

- Not an 8 GiB device (shared memory), so no memory verdict.
- One machine: the chat link is host-to-host on localhost, no NAT or TURN.
- Cancellation, contributor replacement and forced direct-link loss (DO-51 gate 4) are not exercised (WS-28 later tested them on two workers).
- Text parity was not measured: the hashes include the timing line, and no tokens or logits were captured.
- The driver at the time recorded no per-turn transport, so that D's replies came over the data channel rests on the connected status before its turns.
- The original gate script was not kept; `gate23.py` was rewritten from `gate.json`'s fields and reproduces it exactly.

## Reproduce

```bash
# clean export of 377a646 (or later main); bundle: ygaraty.coordinator.prepare_bundle on runs/models/s5-nano --splits 0,32 --capacity 4096 --generate 256
# WORKERS=1 SPLITS=0,32 MEMORY_GB=8, ROUTE=relay then webrtc: runs/experiments/WS-23/omarchy-20261110-1921/go.sh
#   (runs/experiments/WS-22/run.sh with runs/experiments/WS-21/serve.py; paths to the panel and Playwright venvs are local)
.venv/bin/python runs/experiments/WS-23/gate23.py runs/experiments/WS-23/<run-dir>
```

