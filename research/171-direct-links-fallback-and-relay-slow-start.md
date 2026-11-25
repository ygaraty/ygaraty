# Direct links gave a wrong long answer after falling back to the relay; the relay's remaining hop cost is TCP slow start after idle

| | |
|---|---|
| **Conclusion** | Failed gate: direct links gave one wrong long answer and fell back to the relay; D/R 0.83 (measured, 3 RTX 4090) |
| **Status** | executed |
| **ID** | DO-102 (report 171) |
| **Category** | browser |
| **Where** | remote (DigitalOcean coordinator + coturn in fra1; three Vast.ai RTX 4090 24 GB in France) |
| **Priority** | med |
| **Date** | Added 2026-11-24; finished 2026-11-24 |
| **Path** | Report 167's setup with the fixed `proxy.py` (TCP_NODELAY) and `?hopTrace=1` in both arms; blocks R D D R; then one diagnostic R block with `net.ipv4.tcp_slow_start_after_idle=0` on the coordinator |
| **Result** | Item 1 **fails**: block 3 (D) answered the long prompt wrongly from its 4th token ("The user asks: Fact 1: …"); chat and code were right in every block, long right in blocks 1, 2 and 4. Item 2 **fails**: decode at 1k, D 64.7 and 358.1 ms per token against R 257.5 and 252.5 (medians 211.4 vs 255.0, 0.83; bound 0.8); block 2's 64.7 comes from a 1-token request slowed by link set-up, and the hop trace puts both D blocks' cycle at 288–297 ms against R's 341. Item 3 **fails**: both D blocks sent hidden states through the relay for a while (402 and 411 sends, 192 and 227 receive polls, against 235 and about 50 when clean); B's side of A–B opened after 31 s in block 3. **Found** (diagnostic block): with slow start after idle turned off on the coordinator, the relay's transit per token fell from 178 to 71 ms (hidden-state hops 51 and 91 → 7 and 29 ms; the token hop, without a body, unchanged at 36 ms), the cycle from 341 to 228 ms, decode at 1k from 255 to 224 ms |
| **Evidence** | measured-remote: `runs/experiments/DO-102/remote-20261124/` (blocks, serve and contributor logs with `HOP` lines, `hops-*`), diagnostic block in `remote-20261124-ssai0/` |

## Question

Report 170 (WS-52) found that the relay arms of reports 164 and 167 carried about 40 ms per hidden-state hop from the test proxy's Nagle stall, so their relay-versus-direct comparisons were biased against the relay. Measured again without it, and with per-hop timestamps, which route is faster per token, and by how much? Where links fall back to TURN (report 167), is a TURN-relayed data channel slower than the HTTP relay per hop?

## Prediction

TBD. From report 170: relay per hop ≈ one round trip to the coordinator + 3 ms; direct ≈ one peer round trip half-way plus little; TURN ≈ the path through the coordinator. With one-region round trips of 10–15 ms, the relay arm should drop from 216 to about 130 ms per token at 1k.

## Gate (frozen 2026-11-24, before renting)

Blocks R D D R on the same three machines (fast GPUs in one region, coordinator with coturn nearby), report 167's code plus the fixed `proxy.py` and `?hopTrace=1` in both arms.

1. **Same tokens:** chat, code and long give report 156's 64 greedy tokens in every block.
2. **Faster:** D's decode at 1k positions (33-token minus 1-token request) at most 0.8× R's, medians of each arm's valid blocks. A block is invalid if its 1-token request takes longer than its 33-token one; with fewer than one valid block per arm the result is **Inconclusive**.
3. **Links:** in every D block all three pairs open and the coordinator's receive polls stay at the clean level (no hidden states through the relay).

Reported: per-hop transit, compute and read-back from `HOP` lines in both arms; candidate types and round trips.

## Protocol

As report 167 with the fixed proxy; check first, on the workstation, that the fixed proxy gives report 170's 2.9 ms hop (`PROXY=cache` in `WS-52/tools/session.sh`). Budget and teardown deadline from the user first. Prefer hosts without a proxy in front (report 167's 38.x provider stalled loads).

Related: [170 (WS-52)](170-relay-hop-cost-and-proxy-nagle.md), [167 (DO-100)](167-inkling2-fast-stages-relay-vs-turn.md), [164 (DO-97)](164-inkling2-direct-links-three-contributors.md).

## Log

- 2026-11-24: Added at the user's request, from report 170.
- 2026-11-24: User: "run both. budget of 5 approved" (DO-102 and DO-103); about $3 for DO-102, 150-minute watchdogs. Gate frozen above, before renting. Fixed proxy checked locally (report 170: 2.9 ms per hop with `PROXY=cache`).
- 2026-11-24: **Run.** Coordinator DigitalOcean `s-8vcpu-32gb-amd` in fra1 (`do102c`); RTX 4090 24 GB ×3 in France (`do102g1` 82.67.158.138, `do102g2` and `do102g3` 80.15.7.37), 11–21 ms TCP connect to the coordinator. Bundle a3ca405b… (as reports 164 and 167). Contributor proxies with TCP_NODELAY. Blocks 20:28–20:58 UTC.
  - Links: block 2 A–B and A–C srflx (14–15 ms), B–C prflx–relay (39 ms); block 3 A–B and A–C srflx (37–47 ms), B–C relay (86–96 ms). B–C's TURN path made it the slowest hop in both D blocks (99–103 ms transit).
  - Hop trace (medians over 220 decode calls per stage; transit per stage is skewed by host clocks, its sum over a cycle is not): R block 1 transit 178 ms + compute 151 ms + read-back 3 ms per token (cycle 341 ms); D blocks transit 134–137 ms (cycle 288–297 ms). Stage compute on the 4090s: 23, 73–75 and 52–54 ms (A, B, C).
  - Diagnostic block (`remote-20261124-ssai0`, not gated): `sysctl net.ipv4.tcp_slow_start_after_idle=0` on the coordinator only (the contributors' containers refuse it). Transit per token 71 ms, cycle 228 ms, decode at 1k 223.8 ms.
- 2026-11-24: Gate: 1 FAIL, 2 FAIL, 3 FAIL. All machines deleted by their recorded IDs; about **$1.29**.

## Conclusion

Two findings, neither the one the gate asked about. First, the direct path is not safe under fallback: when a link stalls or reopens during a long prefill (hand-offs of about 19 MB per 512-token chunk on Inkling's `f32` wire), hidden states reach the next stage twice or out of order, and the answer goes wrong. Short prompts were unaffected. This needs a local reproduction (a link forced to stall mid-transfer) before direct links are used for long prompts. Second, the relay's remaining per-hop cost is TCP's slow start after idle. Between tokens each connection idles longer than its retransmission timeout, so every 18–37 KB hidden state restarts from a small congestion window and takes two or three round trips. Turning it off on the coordinator removed about 107 ms per token; the contributor side, which containers do not let us change, still pays it on the sending leg. Keeping connections warm (or one long-lived stream per neighbour) would avoid it without a system setting.

## Limits

- Two blocks per arm; block 2's decode figure is unreliable (link set-up during its 1-token request), which the gate's validity rule did not catch.
- The diagnostic block changed only the coordinator's setting, once.
- Stage compute (about 150 ms per token on 4090s) dominates both arms here.

## Reproduce

As report 167 (`runs/experiments/DO-97/tools`), with `PARAMS=hopTrace=1` for `drive.sh` and `runs/experiments/WS-52/tools/hops.py` on each block's contributor logs.
