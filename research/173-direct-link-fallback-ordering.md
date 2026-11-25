# A link dying mid-transfer let later chunks overtake earlier ones and changed the answer; one ordered chain per neighbour fixes it

| | |
|---|---|
| **Conclusion** | Fixed: under injected link stalls, 11 of 11 runs give the relay's answers, against 5 wrong of 9 before (measured, workstation) |
| **Status** | executed |
| **ID** | WS-53 (report 173) |
| **Category** | correctness |
| **Where** | this machine (three native contributors on the 680M, tiny Inkling 2 bundle, direct links) |
| **Priority** | high |
| **Date** | Added 2026-11-24; finished 2026-11-24 |
| **Path** | `qwen35_peer.mjs` `Peers.send` (one chain per neighbour; every message numbered; waiting sends released when a link goes) and `qwen35_contributor.mjs` (`handOn` sends through the peers even when the link is down; `admit` holds a message that arrives ahead of a gap and releases it in order) |
| **Result** | With the fix: every run gives the relay's answers for `long600` (15 chunks) and `chat`, and every stage takes its predecessor's chunks in order (no fault ×2; `stall:3`, `stall:5`, `stall:8` ×3 each); a faulted long request takes about 1.08 s. Before the fix, the same runs: no fault right; `stall:3` wrong 3 of 3, `stall:5` wrong 2 of 3, `stall:8` wrong 0 of 3 (chunks taken out of order in every wrong run); faulted long requests 10.9–15.9 s |
| **Evidence** | measured-local: `runs/experiments/DO-97/omarchy-20261124-local/webrtc-fixed-*` and `webrtc-before-*` (answers, contributor logs with `HOP` lines), `runs/experiments/WS-53/tools/gate.sh` |

## Question

Report 171 (DO-102) saw a three-contributor Inkling 2 session over direct links answer a 2,761-token prompt wrongly from its 4th token, in a block where links fell back to the relay for a while (prefill hand-offs of about 19 MB per 512-token chunk). Short prompts were right. Can a stall, drop or relink during a large hand-off make a stage apply a hidden state twice, skip one, or apply them out of order? If so, what fixes it?

## Prediction

TBD. Suspects: a message partly received on the link and then resent through the relay (de-duplication by `pseq` per link, not per message), acknowledgements lost when a link is replaced, or relay and link deliveries interleaving.

## Gate (frozen 2026-11-24, before the fix)

Tiny bundle, three native contributors on this machine, direct links (`route webrtc`), prompts `long600` (915 tokens, 15 chunks) and `chat`, 48 tokens each, compared with the relay's answers. Fault `?linkFault=stall:N` (test-only): each link's N-th message never drains and the link is declared dead 50 ms later.

- **Fixed** if, with the fix, every run gives the relay's answers: no fault ×2, and N = 3, 5 and 8 ×3 runs each (11 runs); and the hop trace shows each stage taking the previous stage's chunks in order in every faulted run.
- Before the fix, the same runs are reported (how many break).

## Protocol

1. A test-only fault switch (link parameter) in the peer module: stall sends after N bytes, close the channel mid-body, force a relink.
2. Tiny bundle, 3 contributors, a prompt of several chunks; compare with the relay's tokens. Then unit tests for the fixed logic.

Related: [171 (DO-102)](171-direct-links-fallback-and-relay-slow-start.md), [164](164-inkling2-direct-links-three-contributors.md), DO-43/DO-44 (link replacement and stalls).

## Log

- 2026-11-24: Added at the user's request, from report 171.
- 2026-11-24: **Cause found in DO-102's logs.** In report 171's block 3, stage C took B's chunks 128–129 over the link, then over the relay from 130, and at its call 134 a 9-token chunk: the prompt's last chunk (2,761 = 43 × 64 + 9) had overtaken about 30 chunks. Mechanism: when a link is declared dead while its send chain waits in `drained()` for a stalled message, that wait lasts until the channel's close event or `STALL_MS` (5 s); meanwhile `handOn` sees no link and sends the following chunks straight to the relay, without a `pseq` and ahead of the stalled one, and the receiver applies whatever arrives.
- 2026-11-24: **Reproduced locally** (`runs/experiments/DO-97/omarchy-20261124-local/webrtc-ws53fault*`) with `linkFault=stall:5`: in one of three runs stage C took B's last chunk (sent without `pseq`) before chunks 4–13 (`pseq` 5–14), and the long answer differed from the relay's; the other two runs happened to keep the order. No fault: answers equal. Gate frozen above, before the fix.
- 2026-11-24: **Fix.** (1) `handOn` sends every hidden state to the next stage (and the last stage's tokens to the first) through `Peers.send` even when the link is down, so each carries a `pseq` and waits behind the earlier ones; a token the relay carries keeps its `pseq`. (2) `Peers.send` keeps one chain per neighbour instead of per link, so a re-made link cannot overtake relay sends of the old one. (3) Dropping or unlinking a link releases its waiting sends at once (no 5 s wait for a close event that may not come). (4) The receiver takes numbered messages in order: a message ahead of a gap is held (at most 4,096) and released when the gap fills; the ordering step runs before the generation and chat-slot checks, so every numbered message moves the sequence on, and a message parked by those checks is not taken for a duplicate when it comes back. Browser tests 132 pass.
- 2026-11-24: **Gate** (`tools/gate.sh fixed`): PASS, 11 of 11. Same runs on the pre-fix code with only the fault switch and hop trace added (`tools/gate.sh before`, modules copied outside the tree): 5 of 9 faulted runs wrong.
- 2026-11-25 (remote re-check, with WS-54 / report 174): three RTX 4090s in France, direct links with this fix; in both direct blocks links went through TURN and hidden states fell back to the relay for a while (405 and 431 relay sends), and chat, code and the 2,761-token long prompt gave report 156's answers in both (report 171 had one wrong long answer in the same situation).

## Conclusion

Report 171's wrong answer was a real ordering bug, not a hardware fluke. When a link died while a large hand-off was waiting on it, the following hand-offs went straight to the relay and could arrive first, and the receiving stage applied them as they came. Prefill chunks out of order put the prompt's tokens at the wrong positions. The fix keeps one numbered, ordered chain per neighbour and has the receiver restore order. It also makes fallback fast, because a dead link no longer holds its queue for up to 5 seconds.

## Limits

- Local and injected: one fault kind (a stall followed by a drop), the tiny model, host-only links; remotely, two direct blocks with real fallback (log, 2026-11-25), not a forced stall.
- A message that never arrives at all (lost by both paths) now stalls the receiver instead of being skipped; that would show as a turn timing out rather than a wrong answer.

## Reproduce

`bash runs/experiments/WS-53/tools/gate.sh TAG` (uses `runs/experiments/DO-97/omarchy-20261124-local` and its relay answers `relay-ws53`); `BROWSER_ROOT=DIR` runs another copy of the browser modules.
