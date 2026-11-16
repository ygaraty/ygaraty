# Chat turns are bitwise repeatable; WS-23's "mismatches" were the test driver hashing the page's timing line

| | |
|---|---|
| **Conclusion** | Found: chat turns bitwise repeatable (9/9 sessions); WS-23's mismatches were the driver's hash, 15/15 explained (measured) |
| **Status** | executed |
| **ID** | WS-27 (experiment 154) |
| **Category** | correctness |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added and run 2026-11-11 (12:03–12:33 local) |
| **Path** | S5-nano Q4_K_M in headless Chromium on the Radeon 680M, `qwen35_session local --chat` (the coordinator's chat path: prompts arrive as chat turns), driven over HTTP `/api/chat`, ended gracefully so the stages report every call; part 1 arms 1W (`--splits 0,32`) and 2W (`--splits 0,16,32`), 3 turns of 32 tokens with `--logit-digests`; part 2 arm L (1W, WS-23's shape: 3 cycles × 5 turns of 256 tokens with resets, `--record-turn-ids`) |
| **Result** | Part 1: 1W 3/3 and 2W 3/3 sessions identical to their run 1 in prompt ids, tokens, all 96 logit digests and call shapes. Part 2: L 3/3 identical in all 15 turns' prompt ids and tokens (positions up to about 1,500). Cause of WS-23's mismatches: `chat_cycles.py` hashed the reply element after the page had appended the timing line inside it; hashes were equal exactly when the displayed timings were (15/15 turns for H1 against H2 and for D against H2). Driver fixed to hash the reply only; WS-21's, WS-22's and DO-50's "paired runs agree" counts were timing comparisons. Gate: **Found**. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/correctness/chat-repeatability-local-2026-11-11.json`; raw `runs/experiments/WS-27/omarchy-20261111-1203/{main,part2}/` (`gate.json`, per-session `report.json`), WS-23's logs for the cause; scripts `runs/experiments/WS-27/{drive.py,go.sh,gate.py}`; code 4764565 (graceful chat end, `--record-turn-ids`, `done` up to 4 MiB, test `test_a_chat_ends_gracefully_between_turns_only`) with the driver fix in `scripts/experiments/chat_cycles.py` |

## Question

[WS-25](152-decode-repeatability-local.md) showed fixed-prompt decoding is bitwise repeatable, yet [WS-23](150-one-worker-chat-local.md)'s identical chat runs matched on 7 of 15 turns. WS-23's own logs rule out a few causes: prompt lengths and positions are identical run to run, and turn (0, 0) differs between two *fresh* sessions (R and H1, separate browsers), so state left over from earlier conversations cannot be the whole cause. Are chat turns repeatable when compared at the token and logit level, and if not, at which turn and step do they first diverge, and with identical prompt ids?

## Prediction

Unknown; this is a diagnostic. If chat turns are repeatable here, WS-23's mismatches come from what this run leaves out (256-token turns, the chat page and its text hash, panel bundles, resets mid-session).

## Gate (frozen 2026-11-11, before any run)

Diagnostic: the verdict is **Found**.

- Runs: 6 fresh sessions, order 1W 2W 1W 2W 1W 2W. Each: `--chat --capacity 4096 --generate 32 --logit-digests`, three turns with `chat_cycles.prompt(0, t, 500)` for t = 0, 1, 2 (no reset), then a graceful end (SIGINT between turns: every stage gets `stop` and reports `done`).
- Valid session: state `finished`, three turns of 32 tokens, 96 digests from the last stage.
- Per arm, against run 1: prompt ids per turn equal or not; the first diverging (turn, step) for tokens and for digests.
- **Repeatable** (per arm): 3/3 identical prompt ids, tokens and digests. If both arms repeat: the chat path itself is deterministic at this length; next, add back what WS-23 had (256-token turns, the chat page, bundles) one at a time. If not: the first diverging step and its call (prefill or decode) say where to look.

## Protocol

- Code: the working tree at bd3354a plus the graceful chat end (`SessionProtocol.end_chat`; `local` calls it on SIGINT between turns) and WS-26's piece schedule (transport only; relay route here anyway). Model files from `runs/models/s5-nano`, no downloads. Panel venv python.
- Driver `runs/experiments/WS-27/drive.py`; under `watch.py`; one local GPU job at a time; profiles deleted with each run.

## Log

- 2026-11-11: Added at user request ("next 1", the chat-path determinism check after WS-25). Graceful chat end implemented first (stages never reported `done` in chat sessions, so digests and per-call timings were lost); unit test `test_a_chat_ends_gracefully_between_turns_only`. Gate frozen before any run.
- 2026-11-11 12:03, rehearsal (`runs/experiments/WS-27/omarchy-20261111-1203/rehearsal/`, 1W): the graceful end works (stage A reported `done` with 96 calls and 96 digests, state `finished`), but the report's turn records carry counts only. Protocol fix before the main run (gate unchanged): with `--logit-digests` the report also holds `chat_turn_ids` (each turn's prompt ids and tokens; kept out of the chat stream's events). Second rehearsal: turns of 37, 47, 47 prompt tokens, 32 tokens each.
- 2026-11-11 12:07, main run started: `watch.py --stall 300 --max-runtime 1800`, `go.sh RUN "1W 2W 1W 2W 1W 2W"`.
- 2026-11-11 12:11, main run `finished`. Gate by script (`main/gate.json`): **1W 3/3 and 2W 3/3 identical to run 1** in prompt ids, tokens, all 96 logit digests and call shapes. The chat path is bitwise repeatable at 32-token turns. Not the cause either: the split attention (fixed 512-key splits, combined in a fixed order, no atomics), and the matrix-vector kernel (`local` defaults to the same `2,4,8` as bundle sessions, so WS-25 covered it).

## Part 2 gate (frozen 2026-11-11 12:20, before its runs)

As planned above, add back WS-23's turn shape first. Arm L: 1W, `--chat --capacity 4096 --generate 256 --record-turn-ids` (new flag: the same `chat_turn_ids` without digests, which 256-token turns would push past the 32 KiB limit), three cycles of five turns as WS-23 (`chat_cycles.prompt(c, t, 500)`, a reset before each cycle), graceful end. 3 fresh sessions.
- Valid: state `finished`, 15 turns. **Repeatable:** 3/3 identical to run 1 in every turn's prompt ids and tokens. Otherwise report the first diverging (turn, step) and its position.
- Repeatable means the remaining WS-23 differences (the chat page and its text hash, prepared bundles, the panel) are next; not repeatable localizes it to long turns or resets.
- Protocol fix before part 2's runs (gate unchanged): the coordinator refused any contributor `done` report above 32 KiB, so a stage with more than about 350 calls (15 turns of 256 tokens here; any 2048-token run) lost its report and its session could not finish. `done` now has its own 4 MiB limit (`DONE_BYTES`); the other contributor messages keep 32 KiB.

## Part 2 result and the cause

- 2026-11-11 12:13–12:33, part 2 (`part2/`, `watch.py --stall 300 --max-runtime 3600`, `finished`): 3 sessions, 45 turns of 256 tokens. Gate by script: **L 3/3 identical to run 1** in every turn's prompt ids and tokens. Repeatable.
- Cause of WS-23's mismatches, from WS-23's own logs: `chat_cycles.py` hashed the last `.model` element's `textContent` once the controls unlocked, after the page had appended the turn's metadata line ("256 tokens · first after 0.4 s · 92 ms per token") inside that element. Prediction "hashes equal exactly when the displayed timings (first token to 0.1 s, ms per token rounded) are equal" holds on 15/15 turns for H1 against H2 and 15/15 for D against H2. The same driver produced WS-21's, WS-22's and DO-50's "paired runs agree on n of m turns"; those were timing comparisons, not text comparisons.
- Fix: the driver hashes the reply without its `.meta` children (checked in headless Chromium on the page's markup). No engine, coordinator or page change was needed for determinism.
- What this run also added to the code: chat sessions end gracefully (`end_chat`), `done` reports up to 4 MiB, `--record-turn-ids`.

## Results

Gates by script (`gate.py`, recomputed 2026-11-14 from the reports with identical output for `main/` and `part2/`):

| Part | Arm | Valid sessions | Identical to run 1 | What was compared | Repeatable |
|---|---|---|---|---|---|
| 1 | 1W | 3/3 | 3/3 | prompt ids (37, 47, 47 tokens), 96 tokens, 96 logit digests, call shapes | yes |
| 1 | 2W | 3/3 | 3/3 | same | yes |
| 2 | L | 3/3 | 3/3 | 15 turns' prompt ids and 256 tokens each (3,840 calls on the last stage) | yes |

1W and 2W also produce the same prompt ids and tokens as each other; their digests differ (not gated; the 2W stage boundary carries the hidden state in f16).

Cause check on WS-23's logs (`runs/experiments/WS-23/omarchy-20261110-1921/`): per turn, the driver's hashes are equal exactly when the displayed timings (first token to 0.1 s, ms per token) are equal: H1 against H2 7 equal hashes and 7 equal timings, D against H2 8 and 8; the rule holds on 15/15 turns in both pairs. **Found.**

## What it taught

- The served chat path (prefill as chat turns, resets, 256-token turns, one or two stages) is deterministic; together with WS-25, a bitwise-logit gate such as DO-51's is meaningful.
- **Measurement bug:** the driver hashed the reply plus the page's timing line. Every "text parity" count from that driver (WS-21, WS-22, WS-23, DO-50) compared timings. Fixed: the driver strips `.meta` before hashing.
- **Engine gaps found and fixed:** chat sessions never ended gracefully (stages did not report `done`, losing digests and per-call timings); the coordinator refused any `done` report above 32 KiB, so long runs lost their report and could not finish (now 4 MiB, `DONE_BYTES`); turn ids are now recordable without digests (`--record-turn-ids`).

## Limits

- Part 1: 32-token turns, three turns, no reset (digests were capped by the 32 KiB limit when it was frozen).
- Part 2 compares tokens, not logits.
- Neither part used the chat page or prepared panel bundles; the cause check on WS-23 rests on its logs (the coordinator's timings rounded as the page shows them; the page text itself was not kept), not on a rerun of the page with the fixed driver.
- One model and GPU.

## Reproduce

```bash
# src = export of the tree described in the Protocol (or main from 4764565 on); panel venv in drive.py
runs/experiments/WS-27/go.sh runs/experiments/WS-27/<run-dir>/main "1W 2W 1W 2W 1W 2W"
runs/experiments/WS-27/go.sh runs/experiments/WS-27/<run-dir>/part2 "L L L"
.venv/bin/python runs/experiments/WS-27/gate.py runs/experiments/WS-27/<run-dir>/main    # writes gate.json; same for part2
```

