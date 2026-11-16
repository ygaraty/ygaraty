# 2048-token replies and a full 16,384-position conversation run without a stall on the workstation: 32 of 32 turns, capacity reset completes

| | |
|---|---|
| **Conclusion** | Works: 32 of 32 turns at 2048 / 16384, including the capacity reset and early ends (measured, workstation) |
| **Status** | executed |
| **ID** | WS-22 (experiment 149) |
| **Category** | browser |
| **Where** | workstation |
| **Date** | Added and run 2026-11-10 (main runs 18:11–19:44 local) |
| **Path** | S5-nano, 2 stages (0,16,32), both headless Chromium contributors and the coordinator on this machine; WebRTC session route; the real chat page driven by `scripts/experiments/chat_cycles.py`; capacity 16384, reply cap 2048; runs LH (HTTP, long prompts), LD (direct, long), MH (HTTP, mixed short and long) |
| **Result** | LH 10/10, LD 10/10, MH 12/12 turns and every reset completed with Send unlocked; 0 runner or page errors. All three conditions tested: every long turn of LH and LD reached ≥ 1638 tokens (20,358 tokens per run); one capacity reset per run, at turn 7 (position 14,532 plus prompt plus 2048 > 16,384), and the turn carrying it completed; MH's 4 short turns ended by themselves (11–23 tokens, 1.6–2.4 s). Long turns took 181–223 s, about 98 ms per token. Gate: **Works**. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/long-replies-full-context-local-2026-11-10.json`; raw `runs/experiments/WS-22/omarchy-20261110-1806/` (`gate.json`, per-run `events.jsonl`) and `rehearsal-1806/`; code 805cfa9 |

## Question

Today's served cap (256 tokens per reply, 4096 per conversation) cuts most real answers. Before proposing 2048 / 16384 as the default and before renting for [DO-50](DO-50.md) round 2: do 2048-token replies, a conversation that fills its 16384 positions (the capacity reset, never exercised: 0 resets in all 270 turns of [WS-21](148-chat-turns-local-screen.md) and DO-50 round 1), and a mix of short and long replies run without stalls or a locked Send?

## Prediction

All turns complete. Each long run reaches the capacity after 7–9 turns and continues in a new conversation. Turns near the capacity are slower (the full-attention layers read a longer cache), about 3–5 min per 2048-token reply at ~95–120 ms per token.

## Gate (frozen 2026-11-10, before any run)

One bundle (`--capacity 16384 --generate 2048`), one session (route `webrtc`), runs in this order:

| Run | Chat | Prompts | Conversations × turns |
|---|---|---|---|
| LH | HTTP | long (`--words 1500`) | 1 × 10 |
| LD | direct | long (`--words 1500`) | 1 × 10 |
| MH | HTTP | mixed (`--prompts mixed --words 1500`: turns 1, 4, 7, 10 are one-sentence questions) | 1 × 12 |

Driver per-turn bound `--timeout 900`.

- **Pass:** every attempted turn and reset `completed` with Send unlocked (the driver's criterion), no runner errors; LD reaches "WebRTC chat connected".
- **Long output tested:** at least 5 of the 10 turns in each of LH and LD reach 1638 tokens (80% of the cap); otherwise reported as untested, whatever the pass result.
- **Capacity reset tested:** at least one turn in each of LH and LD carries `context_reset` and completes; otherwise reported as untested.
- **Early end tested:** at least 3 of MH's 4 short turns end with `stopped == "end"` (the model finished by itself); otherwise reported as untested.
- Outcome: pass with all three tested → **Works** locally; DO-50 round 2 goes ahead at 2048 / 16384 if approved. Any failure → **Found**: diagnose from the coordinator and browser logs before spending.

## Protocol

- Code: a clean export (`git archive`) of `main` at the commit in the Log, so other sessions' uncommitted work is not in the test; the driver from the working tree (adds `--prompts mixed`).
- Bundle from `ygaraty.coordinator.prepare_bundle` on `runs/models/s5-nano` (no downloads), `--splits 0,16,32 --capacity 16384 --generate 2048`.
- Harness as WS-21 (`runs/experiments/WS-21/serve.py`, coordinator on `localhost:8090` from the panel Python, TURN host with nothing listening), driver in the scratch Playwright 1.65.0 venv with the system Chromium; chat URL only in a mode-600 file and the driver's environment.
- Run under `watch.py` (`--stall 900 --watch` the run directory, `--max-runtime 14400`). Gate computed by script from `events.jsonl`.

## Log

- 2026-11-10: Added at user request ("go ahead") as the local screen for the realistic token limit. Gate frozen above before running.
- 2026-11-10 18:06: clean export of `main` at 805cfa9 into `runs/experiments/WS-22/omarchy-20261110-1806/src`; bundle prepared (16384 / 2048). Driver from the working tree with `--prompts mixed` (sha256 8f23daf3fe78…).
- 2026-11-10 rehearsal (`rehearsal-1806`, HTTP, 1 × 2 mixed): both turns completed; turn 0 gave 1926 tokens in 181 s (93 ms per token) and ended by itself, the short turn 15 tokens in 1.8 s.
- 2026-11-10 18:11: main runs LH, LD, MH started under `watch.py` (`--stall 900 --watch` the run directory, `--max-runtime 14400`).
- 2026-11-10 19:44: main runs finished, **GATE PASS** (`gate.json` in the run directory, computed by script): LH 10/10, LD 10/10, MH 12/12; resets completed; 0 runner or page errors (only `net::ERR_ABORTED`, the page's own abort after each stream). All three conditions tested: every long turn of LH and LD at ≥ 1638 tokens (20,358 tokens each, all 2048 but the first at 1926, which ended by itself); one capacity reset per run at turn 7 (position 14,532 + prompt + 2048 > 16,384), the turn carrying it completed; MH's 4 short turns all ended by themselves (11–23 tokens, 1.6–2.4 s). Long turns took 181–223 s (about 95–105 ms per token, slower near the capacity). LD and MH ran with the driver that also records `submit` events (sha256 685433e9d6e0… after merging the upstream settings-toggle change, which this older page lacks).

## Results

Gate by script (`gate.json` in the run directory, recomputed 2026-11-14 from the events by `runs/experiments/WS-22/gate22.py` with identical output):

| Run | Chat | Turns completed | ≥ 1638 tokens | Capacity reset (completed) | Short turns ending by themselves | Tokens | Max turn | ms per token (long turns) |
|---|---|---|---|---|---|---|---|---|
| LH | HTTP | 10/10 | 10 | 1, turn 7 | — | 20,358 | 222.6 s | 98.5 |
| LD | direct | 10/10 | 10 | 1, turn 7 | — | 20,358 | 208.2 s | 97.7 |
| MH | HTTP, mixed | 12/12 | 8 | 1, turn 10 | 4 of 4 (15, 11, 11, 23 tokens) | 16,322 | 217.6 s | 98.5 |

Resets all completed; the only failed request is the page's own `net::ERR_ABORTED` after each stream. First text visible after a median 0.61 s. The first long turn of each run ended by itself at 1926 tokens; every other long turn hit the 2048 cap. Tested: long output, capacity reset, early end. **GATE PASS: Works.**

## What it taught

- The capacity reset works in the served chat path: the turn that would overflow 16,384 positions starts a new conversation (position back to 2,092 after the turn) and completes normally, on HTTP and direct chat.
- At 16k positions S5-nano slows only slightly (182 s to 223 s per 2048-token turn, peaking just before the reset), so 2048 / 16384 is a workable served cap on this model.
- Speed alone is not a blocker locally: about 98 ms per token with host links. DO-50 round 2 (Phi-4 over a transatlantic TURN link) later ran far slower, explained by WS-24 (two link round trips per decode step).

## Limits

- One machine: no real network between contributors or chat; S5-nano (hybrid attention, small cache) rather than Phi-4, so cache memory and speed at 16k say little about Phi-4.
- One conversation per run, one capacity reset per run.
- LH ran with an older driver than LD and MH (no `submit` events); the gate fields do not depend on it.
- The `gate.py` in `runs/experiments/WS-22/` is a copy of WS-21's (another output format); the original gate computation was not kept as a file, so it was rewritten as `gate22.py` and matches `gate.json` exactly.

## Reproduce

```bash
# clean export of 805cfa9 (or later main); bundle: ygaraty.coordinator.prepare_bundle on runs/models/s5-nano --splits 0,16,32 --capacity 16384 --generate 2048
# harness as experiment 148 (runs/experiments/WS-21/run.sh with serve.py), run list:
cat runs/experiments/WS-22/omarchy-20261110-1806/runs.json
.venv/bin/python runs/experiments/WS-22/gate22.py runs/experiments/WS-22/<run-dir>
```

