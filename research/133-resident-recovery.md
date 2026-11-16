# Chat sessions recover from a cancelled turn and a lost contributor with the resident loop on, with the same tokens

| | |
|---|---|
| **Conclusion** | Works: both faults recover mid-burst, post-fault turns identical, tokens equal to resident-off (measured, workstation) |
| **Status** | executed |
| **ID** | WS-48 (experiment 133) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-13 |
| **Path** | WS-28's part 1 harness on a ONE-stage S5-nano session (`--splits 0,32`), where the contributor runs self-fed turns through the resident loop (experiment 129's addendum) with the new default kernels (experiments 121, 125, 127, 132): `qwen35_session local --chat --capacity 4096 --generate 256 --record-turn-ids --turn-timeout 120`; faults 1.5 s into a turn |
| **Result** | One-stage S5-nano chat sessions, O K C R K C R O. Every completed turn's tokens identical with the resident loop on (K) and off (O). Cancel: the client dropped turn 1 after 1.5 s (22 stream lines in, mid-burst); turn 1 again and turn 2 completed with 256 tokens in both sessions, identical to each other and to K's. Replace: the contributor's Chrome killed 1.5 s into turn 1; the turn ended after 121.7 s with "no token for 120.0 s" (the turn timeout, before the 90 s heartbeat's message reached the client); the replacement brought the session back to `running` in 27.1 and 28.1 s; turn 1 again and turn 2 completed, identical between the sessions and to K's. Resident turns took 13.8 s against 15.3 s with the loop off (256 tokens). **Works.** Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/resident-recovery-2026-11-13.json`; raw `runs/experiments/WS-48/omarchy-20261113-0250/` (`gate.json`, per-session logs and events); code 7aa4df8 |

## Question

The production contributor now runs a one-stage turn as a resident burst that only checks for incoming messages between tokens. Does a session still recover from a client-cancelled turn and from the contributor's browser being killed mid-turn, with the same turns afterwards, and does the resident loop give the same tokens as the step-by-step loop?

## Prediction

Both faults recover as in WS-28; tokens identical to the resident-off control.

## Gate (frozen 2026-11-13, before any run)

Order O K C R K C R O (O: control with `YGARATY_RESIDENT=off`; K: control; C: T0, T1 dropped by the client 1.5 s after it starts, T1 again, T2; R: T0, T1 with the contributor's Chrome killed 1.5 s in, a replacement browser, T1 again, T2). Clean export of the committed code. `runs/experiments/WS-48/gate.py`:
1. **Same tokens:** every completed turn's tokens identical in O-1, O-2, K-1, K-2.
2. **Recovers** (C and R, both sessions): the session ends `finished` with three completed 256-token turns, the turns after the fault complete without error, and the post-fault turns are identical between the two sessions.
3. **Faults land mid-turn:** the faulted turn did not complete its 256 tokens before the fault (otherwise **Inconclusive**: the fault missed the burst).
**Works** if 1–3 hold; **Failed gate** naming the arm otherwise.

## Protocol

Driver `runs/experiments/WS-48/drive.py` (WS-28's, one stage, 256 tokens, 1.5 s, browser0), `go.sh`, `gate.py`; under `watch.py`; one local GPU job at a time. The killed contributor is noticed by its 90 s heartbeat (WS-28's finding), so each R session takes about 2 minutes longer.

## Log

- 2026-11-13: Added at user request ("next top priority?" → this, "yes").

- 2026-11-13 02:50–03:08: run `runs/experiments/WS-48/omarchy-20261113-0250/` (clean export of 7aa4df8; watch.py `finished`, stall 400 s, max 1 h), gate by `gate.py`: **Works**.

## Results

| Session | Turn 0 | Turn 1 | Turn 1 again | Turn 2 |
|---|---|---|---|---|
| O-1, O-2 (resident off) | 256 | 256 | – | 256 |
| K-1, K-2 | 256 | 256 | – | 256 |
| C-1, C-2 | 256 | dropped at 1.5 s | 256 | 256 |
| R-1, R-2 | 256 | killed at 1.5 s; "no token for 120.0 s" at 121.7 s | 256 (running again after 27.1 / 28.1 s) | 256 |

Tokens: O and K identical; post-fault turns identical between repeats and equal to K's turns 1 and 2; every fault landed mid-turn.

## What it taught

- The resident burst hands control back correctly: a cancelled turn and a lost contributor both recover with the same answers as an undisturbed session.
- With one stage, the killed contributor's turn ends on the 120 s turn timeout rather than WS-28's "stage left" message (which came at 93.8 s with two stages): the client waits up to 2 minutes either way; the 90 s heartbeat remains the limit (WS-28's finding).
- The resident loop speeds a one-stage S5-nano chat turn by 10% on the 680M (13.8 against 15.3 s per 256 tokens).

## Limits

- S5-nano and one stage (where the resident loop runs); Qwen3.5-9B and production network conditions not rerun.
- Direct-link loss (WS-28 part 2) not rerun: a one-stage session's direct chat link carries no stage traffic.

## Reproduce

```bash
R=runs/experiments/WS-48/<host-ts>; git archive HEAD | tar -x -C $R --one-top-level=src
python3 <experiments skill>/scripts/watch.py run --dir $R/watch --stall 400 --max-runtime 3600 -- bash runs/experiments/WS-48/go.sh $R "O K C R K C R O"
python3 runs/experiments/WS-48/gate.py $R
```
