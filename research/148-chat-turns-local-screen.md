# Many and long chat turns run without a stall on the workstation once a second-turn stall is fixed: 135 of 135 turns

| | |
|---|---|
| **Conclusion** | Works: 135 of 135 chat turns complete with Send unlocked, after fixing a second-turn stall (measured, workstation) |
| **Status** | executed |
| **ID** | WS-21 (experiment 148) |
| **Category** | browser |
| **Where** | workstation |
| **Date** | Added and run 2026-11-10 (main runs 14:19–15:40 UTC) |
| **Path** | S5-nano, 2 stages (0,16,32), both headless Chromium contributors and the coordinator on this machine; WebRTC session route; the real chat page driven by `scripts/experiments/chat_cycles.py`; HTTP chat and direct chat; caps 256 and 1024. Runs H256, D256, F256 (chat page offline 10 s), R256, H1024, D1024 |
| **Result** | The rehearsal found that every second turn of a conversation stalled ("no token for 120 s") in WebRTC-routed sessions: the first stage kept a finished conversation marked `ended` and dropped its later tokens. With the one-line fix (`ended.delete(conv)` on a new prompt, committed as 1f736d9) every run passes: H256 50/50, D256 50/50, F256 8/8, R256 3/3, H1024 12/12, D1024 12/12 turns and every reset completed; 0 context resets, runner or page errors; 49 of 50 turns per 256 run and 9 of 12 per 1024 run reached 80% of the cap, so long output counts as tested; about 95 ms per token, slowest turn 24.7 s (256) and 99.6 s (1024). Gate: **Works**. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/chat-turns-local-screen-2026-11-10.json`; raw `runs/experiments/WS-21/omarchy-20261110-1505/` (`gate.json`, per-run `events.jsonl`) and `rehearsal-1505/`; gate script `runs/experiments/WS-21/gate.py`; fix 1f736d9 |

## Question

Before renting anything for [DO-50](DO-50.md): do repeated conversations (many turns, resets between conversations) and long replies stall the chat or leave Send locked, on the free local setup? A failure here is cheaper to diagnose than on rented machines.

## Prediction

S5-nano runs cleanly for 256-token turns on both chat transports. Long replies (1024) are the less tested path; capacity resets may appear in the 1024 runs and must be logged, not counted as failures.

## Gate (frozen 2026-11-10, before any run)

Runs, in this order, each a new output directory, one coordinator per bundle:

| Run | Chat | Cap (capacity) | Conversations × turns | Other |
|---|---|---|---|---|
| H256 | HTTP | 256 (4096) | 5 × 10 | |
| D256 | direct | 256 (4096) | 5 × 10 | |
| F256 | HTTP | 256 (4096) | 2 × 4 | chat page offline 10 s on turn 2 (`--fault-turn 2`) |
| R256 | HTTP | 256 (4096) | 1 × 3 | right after F256 on the same session: recovery |
| H1024 | HTTP | 1024 (8192) | 3 × 4 | |
| D1024 | direct | 1024 (8192) | 3 × 4 | |

- **Pass (no-fault runs H256, D256, H1024, D1024):** every attempted turn and reset `completed` (no `error`, no `blocked`, no `runner-error`), with Send unlocked at the end of each turn (the driver's criterion). Direct runs must reach "WebRTC chat connected" (otherwise the run fails; a fallback is not a direct pass).
- **Fault (F256):** the faulted turn ends bounded: `completed` or `error` with controls unlocked, within the driver's 150 s; no `blocked`. **R256** must complete all its turns (recovery).
- **Long output:** the 1024 condition counts as tested only if at least 4 of the 12 turns per 1024 run reach 819 tokens; otherwise it is reported as untested, whatever the pass result. Same rule at 256: at least 10 turns per 256 run reach 205 tokens.
- Context-capacity resets (`context_reset` in the events) are counted and reported, not failures.
- Outcome: all pass → **Works** locally, and DO-50's rented run goes ahead. Any failure → **Found**: diagnose from the coordinator and browser logs before spending.

## Protocol

- Code: a clean export of `main` at the commit in the Log (`git archive`), so the uncommitted work of other sessions is not in the test; the driver from the working tree with its local-run changes (localhost URL, `--browser-executable`, `context_reset` recorded).
- Bundles from `ygaraty.coordinator.prepare_bundle` on `runs/models/s5-nano` (no downloads): `--capacity 4096 --generate 256` and `--capacity 8192 --generate 1024`.
- Coordinator on `localhost:8090` with the panel database in the run directory, WebRTC enabled, a TURN host with nothing listening (host candidates on one machine), as `runs/checks/*/run.sh`. One session (route `webrtc`), two contributors, the chat URL written to a mode-600 file and passed to the driver in the environment, never printed.
- Driver: Playwright 1.65.0 in a scratch venv with the system Chromium 155, `--words 500`, default 150 s per-turn bound; each run under a 2 h `timeout`.

## Log

- 2026-11-10: Added at user request ("test local") as DO-50's local screen. Gate frozen above before running.
- 2026-11-10 15:05: clean export of `main` at 620d13c into `runs/experiments/WS-21/omarchy-20261110-1505/src`; bundles prepared (256/4096, 1024/8192).
- 2026-11-10 rehearsal (`rehearsal-1505`, 1 × 2 turns per transport), three findings before the main runs:
  1. The coordinator needs the panel Python (FastAPI); the repository `.venv` has none.
  2. **Driver bug:** the chat page aborts its own `/api/chat` request when the stream ends (b20149c's cleanup), after which Playwright cannot read the response body, so every turn was recorded `stream-read-failed`. Fixed: the driver now reads the stream inside the page (fetch wrapped, body teed); also records `context_reset`, the page's error text and failed request paths (no query or key), accepts a localhost URL and `--browser-executable`. Driver sha256 5b1d1b5156d3…
  3. **Found the stall:** with the driver fixed, turn 0 completed (256 tokens, 24 s) and **turn 1 of the same conversation failed with "no token for 120 s"** on both HTTP and direct chat. Cause: in WebRTC-routed sessions the first stage marks a conversation `ended` at its last token and drops later tokens for it (added by 4993ca6, DO-25 work), and a new turn's prompt never cleared the mark, so every second turn stalled. One-line fix in `qwen35_contributor.mjs` (`ended.delete(conv)` on a new prompt); rehearsal with the fix: 6/6 HTTP turns (2 × 3) and 3/3 direct turns completed.
- 2026-11-10 pre-run amendment (before any main run): main runs use 620d13c + that one-line fix (patched file sha256 fb1141c7cf47…); the 1024 runs use `--timeout 300`, since a 1024-token reply alone takes ~100 s at the measured 95 ms per token. Gate otherwise unchanged.
- 2026-11-10 15:2x: main runs started (`omarchy-20261110-1505`) under `watch.py` (`--stall 600 --watch` the run directory, `--max-runtime 10800`).
- 2026-11-10: main runs finished, **GATE PASS**: H256 50/50, D256 50/50, F256 8/8, R256 3/3, H1024 12/12, D1024 12/12; 0 context resets, page or runner errors; 49 of 50 turns per 256 run and 9 of 12 per 1024 run at ≥ 80% of the cap; slowest turn 24.7 s (256) and ~100 s (1024). The 10 s offline window did not break the open localhost stream, so F256 tests little. DO-50's rented Phi-4 round confirmed the same with the fix and reproduced the stall without it.

## Results

Gate by script (`gate.py`, recomputed 2026-11-14 from the events with identical output):

| Run | Chat | Cap | Turns completed | Resets | ≥ 80% of cap | Ended early | Max turn | ms per token |
|---|---|---|---|---|---|---|---|---|
| H256 | HTTP | 256 | 50/50 | 5/5 | 49 | 1 | 24.7 s | 95.5 |
| D256 | direct | 256 | 50/50 | 5/5 | 49 | 1 | 24.7 s | 95.9 |
| F256 | HTTP, 10 s offline on turn 2 | 256 | 8/8 | 2/2 | 8 | 0 | 24.5 s | 94.8 |
| R256 | HTTP (recovery) | 256 | 3/3 | 1/1 | 3 | 0 | 24.3 s | 94.6 |
| H1024 | HTTP | 1024 | 12/12 | 3/3 | 9 | 4 | 99.6 s | 94.7 |
| D1024 | direct | 1024 | 12/12 | 3/3 | 9 | 4 | 97.8 s | 94.7 |

0 context resets, 0 runner errors, 0 page errors in every run; the only failed request each run is the page's own `net::ERR_ABORTED` on `/api/chat` when a stream ends (the page aborts the finished request; expected). First text visible after a median 0.61 s. HTTP and direct runs generated the same token counts (12,642 at 256, 11,357 at 1024). Long output tested in all four no-fault runs (needed 10 and 4). **GATE PASS: Works.**

## What it taught

- **Bug found and fixed:** in WebRTC-routed sessions the first stage marked a conversation `ended` at its last token (4993ca6, DO-25 work) and never cleared the mark on a new prompt, so every second turn of every conversation stalled. One line in `qwen35_contributor.mjs` fixed it (1f736d9); DO-50's rented Phi-4 round reproduced the stall without the fix and passed with it.
- **Driver bug:** the page aborts its own `/api/chat` request when the stream ends, so Playwright could not read the body; the driver now reads the stream inside the page.
- The coordinator needs the panel environment (FastAPI); the repository `.venv` lacks it.
- A 1024-token reply takes about 100 s at 95 ms per token, so a 150 s per-turn bound is too tight once prompts grow; 1024 runs used 300 s.

## Limits

- One machine: no real network between contributors or chat, so it cannot show NAT/TURN failures; S5-nano rather than Phi-4.
- The offline fault only cuts the chat page, not the stage links, and the 10 s window did not break the open localhost stream: F256 tests little.
- No context-capacity reset was reached (0 in 135 turns), so that path is untested here (WS-22 tests it).
- The pre-fix stalled rehearsal turns are not kept: `rehearsal-1505/` holds the post-fix rerun (6/6 HTTP, 3/3 direct); the stall is recorded in the Log and was reproduced by DO-50.

## Reproduce

```bash
# clean export of 620d13c plus the ended fix (or any later main), bundles via ygaraty.coordinator.prepare_bundle on runs/models/s5-nano
# (--capacity 4096 --generate 256, --capacity 8192 --generate 1024); panel Python with FastAPI in PANEL_PY
runs/experiments/WS-21/run.sh runs/experiments/WS-21/<run-dir> <profiles-dir> <playwright-python> \
  "$(cat runs/experiments/WS-21/omarchy-20261110-1505/runs-256.json)" "$(cat runs/experiments/WS-21/omarchy-20261110-1505/runs-1024.json)"
.venv/bin/python runs/experiments/WS-21/gate.py runs/experiments/WS-21/<run-dir>
```

