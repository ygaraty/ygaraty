# A 100 ms recv wait keeps relay connections out of slow start: 22% less time per token for Inkling 2's three contributors, no system setting

| | |
|---|---|
| **Conclusion** | Works: answering idle recvs every 100 ms cuts relay transit 150 → 85 ms per token, cycle −22% (measured, local and 3 RTX 4090) |
| **Status** | executed |
| **ID** | WS-54 (report 174) |
| **Category** | transport |
| **Where** | this machine (network-namespace emulation) and remote (DigitalOcean coordinator in fra1, three Vast.ai RTX 4090 in France, default TCP settings) |
| **Priority** | med |
| **Date** | Added 2026-11-24; finished 2026-11-25 |
| **Path** | Coordinator `recv` long-poll: `YGARATY_RECV_WAIT_MS` (new; default unchanged at 20 s). At 100, an idle contributor's `recv` is answered empty every ~100 ms and asked again, so the connection that carries each hidden state to the next stage never idles past TCP's retransmission timeout |
| **Result** | Local (12.3 ms round trip, MSS 1448, emulated 80 ms compute): hidden-state hops 27.3–27.5 ms (2.22 × the round trip) by default, 15.3–15.5 ms (1.25 ×) with the 100 ms wait; answers unchanged. Remote (S = 60.8 ms, the three contributors' connect times summed; bound 1.3 S + 9 = 88.0 ms): total transit per token R 148.4 and 151.8 ms, W 84.2 and 85.8 ms (median 85.0, within the bound); token cycle R 311–326 ms, W 242–248 ms; chat, code and long as report 156 in every block. Cost: about 4,700 `recv` answers per block against about 800 |
| **Evidence** | measured-local: `runs/experiments/WS-54/local/relay-gate-*`; measured-remote: `runs/experiments/WS-54/remote-20261125/` (`gate_remote.py` output, blocks with `HOP` logs, `connect.txt`) |

## Question

Report 171 (DO-102) found TCP slow start after idle costs the relay about 107 ms per token for Inkling 2's three contributors: between tokens each connection idles past its retransmission timeout, so each 18–37 KB hidden state restarts from a small congestion window and takes two or three round trips. Turning the setting off on the coordinator removed most of it, but contributors in containers cannot change it. Can the contributor and coordinator avoid it themselves?

## Prediction

TBD. A connection that never idles past the RTO keeps its window; per-hop transit should approach one round trip, as with the setting off.

## Gate (frozen 2026-11-24, local part)

Local emulation (`runs/experiments/WS-54/tools/netns.sh`): a private network namespace, loopback delay 6 ms each way (round trip about 12.3 ms), routes pinned to MTU 1500 (MSS 1448), `tcp_slow_start_after_idle` at its default (1); a 2,304-wide tiny Inkling 2 (`tools/tiny_wide.py`, f32 hand-offs of real size), three native contributors on the relay with `computeMs=80` (cycle about 300 ms, as on rented GPUs) and `hopTrace`. Arms: **base** (recv waits 20 s) and **warm** (`YGARATY_RECV_WAIT_MS=100`); two runs each, chat, code and multi.

- **Works (local)** if warm's median hidden-state hop transit (stages B and C) ≤ 1.3 × the round trip, base's ≥ 1.8 × (the effect is present), and every run's answers equal the relay's without emulation.
- **Remote check (frozen 2026-11-24, before renting; user's budget $2.50):** three rented GPUs and a DigitalOcean coordinator in one region, default TCP settings everywhere, relay route, `hopTrace`; blocks R W D D W R (R: recv waits 20 s; W: `YGARATY_RECV_WAIT_MS=100`; D: direct links with report 173's ordering fix, a re-check of report 171). Per block, total transit per token from the hop trace (host clock offsets cancel over a cycle) against the sum S of the three contributors' TCP connect times to the coordinator. **Works (remote)** if W's median total transit ≤ 1.3 × S + 9 ms, R's median is above it, and chat, code and long give report 156's tokens in every block. Reported: decode at 1k per block, D's links, fallback and tokens.

## Protocol

1. Local emulation that reproduces the idle reset (a network namespace with netem delay, where slow start after idle applies), then the change.
2. A remote check on rented machines (relay only), with the setting left at its default.

Related: [171 (DO-102)](171-direct-links-fallback-and-relay-slow-start.md), [170 (WS-52)](170-relay-hop-cost-and-proxy-nagle.md).

## Log

- 2026-11-24: Added at the user's request, from report 171.
- 2026-11-24: **Local emulation built.** Loopback's MTU is 65,536 (initial window 640 KB), so slow start never showed; pinning the loopback routes to MTU 1500 / MSS 1448 and adding 80 ms of emulated compute per call (`?computeMs`, test-only) reproduces it: hidden-state hops 27.6 ms (about 2 round trips + 3 ms) with the default, 15.3 ms with the setting off; token hops 15 ms either way. A coordinator that answers an idle `recv` after 100 ms (`YGARATY_RECV_WAIT_MS=100`, new knob; default unchanged at 20 s) gave 15.4 and 15.2 ms with the default setting. Gate frozen above, before the formal runs.
- 2026-11-24: **Local gate passes** (`runs/experiments/WS-54/local/relay-gate-*`, reference answers `relay-ref` without emulation): base hidden-state transit 27.3–27.5 ms (2.22 × the 12.3 ms round trip; token hops 15.0 ms; cycle 335 ms); warm 15.3–15.5 ms (1.25 ×; token hops 14.8 ms; cycle 308 ms); answers equal the reference in all four runs. Cost of the warm arm: each contributor's `recv` is answered about ten times a second while idle. The remote check (rented machines, default settings, `YGARATY_RECV_WAIT_MS=100` on the coordinator) is still to run and needs a budget; WS-54 stays ongoing until then.
- 2026-11-24/25: **Remote check** (budget $2.50): coordinator `ws54c` (DigitalOcean `s-8vcpu-32gb-amd`, fra1, `tcp_slow_start_after_idle` = 1); `ws54g1` RTX 4090 90.39.222.230, `ws54g2` and `ws54g3` RTX 4090 80.15.7.37 (France). Connect times to the coordinator 17.6, 21.3 and 21.7 ms (S = 60.8 ms). Blocks R W D D W R, 23:03–23:50 UTC, bundle a3ca405b…, fixed proxy, `hopTrace`.

  | Block | Total transit per token (ms) | Cycle (ms) | Stage compute (ms) | Decode at 1k (ms/token) | Tokens | Relay sends / recvs |
  |---|---:|---:|---:|---:|---|---:|
  | 1 R | 148.4 | 326.5 | 156.5 | 220.1 | right | 826 / 795 |
  | 2 W | 84.2 | 242.2 | 153.7 | 232.4 | right | 824 / 4,643 |
  | 3 D | 185.7 | 345.6 | 158.9 | 263.6 | right | 405 / 226 |
  | 4 D | 185.4 | 346.8 | 160.0 | 622.4 | right | 431 / 260 |
  | 5 W | 85.8 | 247.6 | 156.1 | 274.8 | right | 824 / 4,736 |
  | 6 R | 151.8 | 311.5 | 158.5 | 335.9 | right | 824 / 817 |

  The decode-at-1k figure (33-token minus 1-token request) varied with each request's prefill and is not used; the hop trace's per-step cycle is. D blocks: all links through TURN (39–48 ms), B's side of A–B opening after 42–45 s, relay fallback for a while; answers right (report 173's fix, see there). Machines deleted by their recorded IDs after 0.87 h, about **$1.50**.
- 2026-11-25: **Gate**: local PASS; remote PASS (W 85.0 ≤ 88.0 ms; R 150.1 > 88.0; tokens right in every block).

## Conclusion

The relay's remaining per-hop cost was TCP's slow start after idle on the coordinator's answers, and the coordinator can avoid it on its own: answering idle receives every 100 ms keeps those connections active, so each hidden state goes out in one flight. On three RTX 4090s in one region that cut transit from about 150 to 85 ms per token, and the per-token cycle by 22%, with the same answers and without touching any machine's settings. The price is about ten empty answers a second per idle contributor. The remaining relay transit (85 ms over three hops at about 20 ms connect time each) is close to its floor; stage compute (about 155 ms per token on 4090s) is now the larger share.

## Limits

- One region, one rental, two blocks per arm; one local emulation (fixed delay, no loss).
- The contributors' sending leg was not changed; with these hand-off sizes it did not show a cost.
- Not yet the default: `YGARATY_RECV_WAIT_MS` stays unset unless chosen; the extra polling should be weighed against coordinators with many contributors.

## Reproduce

Local: `COMPUTE_MS=80 [RECV_WAIT_MS=100] bash runs/experiments/WS-54/tools/netns.sh runs/experiments/WS-54/local 6 1 TAG` (bundle from `tools/tiny_wide.py --hidden 2304`). Remote: DO-97's tools with blocks `"R W D D W R"` (W sets the wait on the coordinator), then `python3 runs/experiments/WS-54/tools/gate_remote.py OUT`.
