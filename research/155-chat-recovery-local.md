# A chat session recovers from a cancelled turn, a killed contributor and a broken direct link, with turns identical to an undisturbed session

| | |
|---|---|
| **Conclusion** | Works: all three faults recover, every later turn identical to an undisturbed session (4/4 and 2/2 sessions, measured) |
| **Status** | executed |
| **ID** | WS-28 (experiment 155) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added and run 2026-11-11 (14:32–14:57 local) |
| **Path** | Part 1: S5-nano Q4_K_M, two stages (`--splits 0,16,32`), `qwen35_session local --chat --record-turn-ids`, 64-token turns over HTTP `/api/chat`; arms K (control), C (the client drops turn 1's stream after 2 s), R (stage B's Chrome killed with SIGKILL 2 s into turn 1, a replacement browser joins); order K C R K C R. Part 2: one-worker panel session (`s5-nano-1w`, 256-token turns) on the `private` route, direct chat through a local TURN server that is killed 3 s into turn 1 (X) or left alone (C); order X C X, run twice (the second with transport recorded) |
| **Result** | C: the dropped turn ended at once; T1 again and T2 completed (64 tokens) and were identical between the two sessions and to K's tokens. R: the killed stage's turn ended after 93.8 s ("stage B left; the conversation resumes when a browser takes its place"), the session was `running` again 16.0 and 17.0 s after the replacement started, and T1 again and T2 were identical between sessions and to K's tokens. Part 2: in both X sessions turn 1 got 31 and 34 of 256 tokens over WebRTC, the rest from the coordinator, the page switched to its fallback, and all three turns' reply hashes equal C's. Gate: **Works**. Finding: a crashed contributor is noticed only after the 90 s heartbeat timeout. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/chat-recovery-local-2026-11-11.json`; raw `runs/experiments/WS-28/omarchy-20261111-1432/main/` (`gate.json`, per-session `report.json` and `*.events.jsonl`), `runs/experiments/WS-28/part2-1445/` and `part2b-1451/` (`*/T/events.jsonl`, `fault.txt`); scripts `runs/experiments/WS-28/{drive.py,go.sh,go2.sh,gate.py}`; code b1bcc5b; driver's transport recording 22bcdec |

## Question

[DO-51](DO-51.md)'s lifecycle gate (cancel, contributor replacement, direct-link loss) has never run on real browsers. After each fault, does the session return to serving, do the following turns complete, and are they repeatable? (WS-27 established that undisturbed chat turns are bitwise repeatable, so equality between repeats is a sharp check.)

## Prediction

Cancel and replacement both drop the stages' state; the next turn replays the conversation (one prefill of the history), so its tokens are repeatable between repeats, but may differ from an undisturbed session, where the same history went through one decode step per token (different matrix kernels).

## Gate, part 1 (frozen 2026-11-11, before any run)

- Sessions: fresh `local --chat` sessions, `--capacity 4096 --generate 64 --turn-timeout 120`, prompts `chat_cycles.prompt(0, t, 500)`, graceful end. Arms, order K C R K C R:
  - K (control): T0, T1, T2.
  - C (cancel): T0; T1, whose stream the client closes 2 s after it starts; T1 again; T2.
  - R (replace): T0; T1, during which browser 1's Chrome is killed with SIGKILL 2 s after it starts; a replacement browser joins with the session's join URL; once the session is `running` again (at most 300 s), T1 again; T2.
- **Recovers** (per arm C and R, both sessions): the session returns to `running`; the turns after the fault (T1 again, T2) complete with 64 tokens and no error; and the two sessions' post-fault turns are identical in prompt ids and tokens. The faulted turn's outcome (error text, tokens before the fault) is reported, not judged.
- Descriptive: post-fault turns against K's T1 and T2 (equal, or the first diverging step); time from the replacement's start to `running`.
- **Works** if C and R both recover; **Failed gate** naming the arm otherwise (a hang, an error after recovery, a failed session, or differing repeats).

## Protocol

- Code: `main` at b1bcc5b (clean export). Model files from `runs/models/s5-nano`; panel venv. Driver `runs/experiments/WS-28/drive.py`; under `watch.py`; one local GPU job at a time.

## Log

- 2026-11-11: Added at user request (item 2 of "what is next?"); part 1 gate frozen before any run.
- 2026-11-11 14:32, rehearsal (`runs/experiments/WS-28/omarchy-20261111-1432/rehearsal/`, R): works end to end; the coordinator noticed the killed stage 92 s after the kill (the roster's 90 s heartbeat timeout).
- 2026-11-11 14:36–14:52, part 1 main (`main/`, `watch.py --stall 400 --max-runtime 2700`, `finished`), K C R K C R. Gate by script (`main/gate.json`): **Works.** C: the dropped turn ended at once, T1 again and T2 completed (64 tokens, no error) in both sessions, identical between them. R: the killed stage's turn ended after 93.8 s with "stage B left; the conversation resumes when a browser takes its place"; the session was `running` again 16.0 and 17.0 s after the replacement started; T1 again and T2 completed, identical between the sessions. Descriptive: post-fault tokens equal K's T1 and T2 in both arms (the replayed T1's prompt holds the history, so its prompt ids differ, its tokens do not). K's two sessions identical.
- Finding: a crashed contributor is noticed only when its heartbeat expires, 90 s, during which the turn waits; deliberate (background tabs' throttled timers), but a closed long-poll connection mid-turn could signal it faster. Not changed here.

## Gate, part 2 (frozen 2026-11-11 14:55, before its runs)

- Sessions: one-worker S5-nano panel sessions (WS-23's `s5-nano-1w` bundle, capacity 4096, generate 256) on the `private` route, so the chat browser's direct link runs only through the local TURN server (WS-24's, no delay); `chat_cycles.py --arm direct --cycles 1 --turns 3 --words 500 --token-cap 256 --timeout 600` (direct chat connected before the first turn). Order X C X. X: the TURN server is killed 3 s after turn 1 is submitted (the direct link's only path); C: undisturbed.
- **Recovers** (both X sessions): all three turns complete (256 tokens, no error, none `blocked`), and every turn's reply hash (reply only, WS-27's driver) equals C's: no missing or duplicated text after the link broke.
- **Works** if both X sessions recover; otherwise **Failed gate** with the turn and outcome.
- 2026-11-11 14:45–14:50, part 2 (`part2-1445/`, `watch.py`, `finished`), X C X: every turn completed (256 tokens, no error) in all three sessions, and the reply hashes of turns 0, 1 and 2 are identical in X-1, C-1 and X-2 (TURN killed at 14:46:31 and 14:49:46, 3 s after turn 1's submit). By the frozen criteria: recovers. But the run cannot show that the direct link carried the turns and then broke: the driver records neither the page's per-reply WebRTC count nor its transport status, and the coordinator does not log a turn's prompt transport. So it is not yet evidence for direct-link loss. Instrumented the driver (`meta`: the reply's own metadata line, "WebRTC: prompt + n/256 tokens" or "prompt via coordinator"; `transport`: the page's status) and rerun as a confirmation (`part2b-*`), same gate.
- 2026-11-11 14:51–14:57, part 2 confirmation (`part2b-1451/`, `finished`), X C X: **recovers, with evidence.** X-1 and X-2: turn 0 "WebRTC: prompt + 256/256 tokens"; TURN killed 3 s into turn 1 (14:52:35, 14:55:52): turn 1 "WebRTC: prompt + 31/256" and "+ 34/256", the rest from the coordinator's stream, completed; the page switched to "Using coordinator fallback · The first contributor link failed: peer connection disconnected"; turn 2 "WebRTC: 0/256 · prompt via coordinator", completed. C-1: all three turns "prompt + 256/256" over WebRTC. Reply hashes of turns 0, 1 and 2 identical in all three sessions (and to the first part 2 run): no missing or duplicated text across the switch. Gate: **Works.**

## Results

Part 1 gate by script (`gate.py`, recomputed 2026-11-14 with identical output): control repeatable (K-1 = K-2).

| Arm | Faulted turn (both sessions) | Back to `running` after replacement | T1 again, T2 | Identical between repeats | Against K's T1, T2 |
|---|---|---|---|---|---|
| C (cancel) | ended after 2.05 / 2.08 s, no error | — | 64 + 64 tokens, no error | yes | tokens equal (T1's prompt holds the history, so its prompt ids differ) |
| R (replace) | ended after 93.8 / 93.8 s, "stage B left; …" | 16.0 / 17.0 s | 64 + 64 tokens, no error | yes | tokens equal (same note) |

Verdict: C and R both recover: **Works**.

Part 2 (reply-only hashes, WS-27's driver; TURN killed 14:46:31 and 14:49:46 in the first run, 14:52:35 and 14:55:52 in the confirmation):

| Session (confirmation) | Turn 0 | Turn 1 | Turn 2 | Reply hashes equal to C-1 |
|---|---|---|---|---|
| X-1 | WebRTC: prompt + 256/256 | WebRTC: prompt + 31/256, rest via coordinator | 0/256, prompt via coordinator | 3/3 |
| C-1 | prompt + 256/256 | prompt + 256/256 | prompt + 256/256 | — |
| X-2 | WebRTC: prompt + 256/256 | WebRTC: prompt + 34/256, rest via coordinator | 0/256, prompt via coordinator | 3/3 |

Every turn completed with 256 tokens in all six part 2 sessions; the first run's hashes equal the confirmation's turn for turn. After the kill the page showed "Using coordinator fallback · The first contributor link failed: peer connection disconnected". **Works.**

## What it taught

- Cancel and replacement both rebuild the stages' state by replaying the conversation as one prefill, and the following turns still give the undisturbed session's tokens: the replayed prefill and the step-by-step decode agree on this model (the prediction allowed them to differ).
- A broken direct chat link falls back to the coordinator's stream mid-reply without missing or duplicated text.
- **Finding:** a crashed contributor is noticed only when its heartbeat expires (90 s; about 92 s after the kill in the rehearsal), and the turn waits that long. Deliberate (background tabs' throttled timers), but a closed long-poll connection could signal a crash sooner (WS-29 asks how long a departed contributor holds its slot).
- **Driver gap:** the first part 2 run passed by its criteria but could not show that the direct link carried the turns; the driver now records the reply's metadata line and the page's transport status.

## Limits

- Part 1 has no direct chat (`local --chat` uses the relay); 64-token turns; two sessions per arm.
- Part 2 is one worker on localhost; the TURN kill breaks the only path, not a lossy one.
- One machine; the replacement browser runs on the same host as the rest. Since experiment 129 production uses the resident decode, and this recovery has not been rerun with it.

## Reproduce

```bash
# part 1: clean export of b1bcc5b in <run-dir>/src; panel venv in drive.py
runs/experiments/WS-28/go.sh runs/experiments/WS-28/<run-dir> "K C R K C R"
.venv/bin/python runs/experiments/WS-28/gate.py runs/experiments/WS-28/<run-dir>    # writes gate.json
# part 2: <run-dir2> holds src/ (b1bcc5b) and models/s5-nano-1w (WS-23's bundle); WS-24's TURN built
runs/experiments/WS-28/go2.sh runs/experiments/WS-28/<run-dir2> "X C X"
```

