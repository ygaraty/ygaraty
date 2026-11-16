# With close stages, direct links cut user-session time per token 60%; one private session in eight failed to link

| | |
|---|---|
| **Conclusion** | **Failed gate:** direct links cut time per token 60% (287 → 115 ms), but one private session never linked (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-44 |
| **Date** | 2026-11-09 |
| **Path** | As [experiment 98](098-user-sessions-webrtc-routes.md) (user-panel path, S5-nano two-stage bundle, chat through `/v1/chat/completions`, coordinator with coturn in DigitalOcean `ams3`), with 98's negotiation fixes (3e80cd9) and both GPUs in Poland on different machines: stage A an RTX 3060, stage B an RTX A2000, 14–18 ms apart, each about 30 ms from the coordinator. Arms R relay, W direct, P private (TURN only) |
| **Result** | **Speed and correctness passed; link reliability failed.** All 9 sessions (R W P / P W R / W R P) gave identical texts for both gated requests; the 8-token limit held; no disagreement; no address in any route record. Median time per token (first request): R 287 ms, **W 115 ms (−60%**, gate −25%), **P 284 ms (−1%**, gate ≤ R). Every W link and 2 of 3 P sessions linked at once (first token 0.8–1.5 s). In P session 9 neither stage's link opened: stage A reported only after 70 s, the first token came after 70 s and the turn took 225 s (52 tokens, median 323 ms, with long stalls). Three more P sessions run afterwards as a diagnostic, outside the gate, all linked at once. Cost about $0.15 of the $1 budget |
| **Evidence** | Measured: [evidence/browser/user-sessions-webrtc-close-stages-2026-11-09.json](evidence/browser/user-sessions-webrtc-close-stages-2026-11-09.json) (gate, per-session times and links, diagnostic sessions); raw results and session logs in `runs/experiments/DO-44/` (not in git). Code 3e80cd9 and 8280605, nothing copied by hand. Gate `runs/experiments/DO-44/gate_remote.py`; launcher `runs/experiments/DO-44/launch_remote.py` |

## Question

Experiment 98 failed its gate: direct links were only 11% faster with the stages 45–52 ms apart, and a
negotiation race (fixed after the run) left 2 of 6 links on the relay and made 4 of 6 sessions wait 30 s.
With the fixes, do every W and P link open at once, and with the stages close together, do direct links
give the large cut experiment 97 saw on the experiment path?

## Prediction

Every link opens within a second or two of `go`. W well below R (experiment 97: −61% with 2–3 ms
between the stages); P between W and R, since every handoff still goes to Amsterdam and back.

## Gate

Frozen 2026-11-09 12:46 CET, before any run.

- Two Vast.ai GPU hosts on different machines (different public addresses) in the same country; the
  coordinator as in experiment 98. Arms R, W, P, nine sessions in the order R W P / P W R / W R P, each
  with a 23-token prompt (`max_tokens` 128), the conversation continued (`max_tokens` 8), and a new
  prompt (`max_tokens` 32, not gated).
- **Correctness:** each gated request's text identical across all nine sessions; the second request
  at most 8 tokens; no `disagreed` line; no IPv4 address in any route record.
- **Fixes work:** in every W and P session both stages report their link open (P: `relay`–`relay`
  on both ends), and the first request's first token comes within 5 s.
- **Works** if W's median time per token (first request, median of its 3 sessions) is at least 25%
  below R's, and P's at most R's.
- A failure of any line fails the gate; a link that fell back still counts as its arm and is labeled.
- Computed by `runs/experiments/DO-44/gate_remote.py` (experiment 98's gate plus the first-token and
  all-links-open checks).

## Protocol

As experiment 98's remote part (`remote.sh panel-coordinator-setup`, `turn-serve`,
`panel-coordinator-serve`, `panel-check`; `launch_remote.py` opens each link on its stage's host),
from commit 8280605, nothing copied by hand.

## Log

- 2026-11-09 12:46 CET: added; gate frozen. Budget **$1**, teardown deadline **14:45 CET** (2 h),
  approved by the user.
- 12:48: pools do44a (Poland, RTX 3060), do44b (Poland, RTX A2000, another machine), do44coord (`ams3`);
  watchdogs 110–115 min.
- 12:50: setup done; both GPUs 30 ms from the coordinator; only the hardened coturn running (the fixes of
  experiment 98 held: no package service, config read).
- 12:57: rehearsal W, P passed: W `srflx`–`srflx` 16–17 ms, P `relay`–`relay` 59–63 ms, links open at
  `go`, first token 1.1–1.7 s. Main matrix started.
- 13:11: main matrix done; gate **failed** on session 9 (P). Diagnostic outside the gate: three more P
  sessions with verbose coturn logging: all linked at once.
- 13:14: all three pools deleted; `pods ls`: 0 machines.

## Results

Time per token of the first request (every session generated 52 tokens, ending at `<|im_end|>`):

| # | Arm | ms per token | First token (s) | Links (round trip) |
|---|---|---|---|---|
| 1 | R | 283 | 0.98 | — |
| 2 | W | 114 | 0.83 | `srflx`–`srflx` UDP 18 / 18 ms |
| 3 | P | 279 | 1.49 | `relay`–`relay` UDP 60 / 61 ms |
| 4 | P | 284 | 1.14 | `relay`–`relay` UDP 63 / 62 ms |
| 5 | W | 115 | 1.01 | `srflx`–`srflx` UDP 14 / 15 ms |
| 6 | R | 287 | 0.93 | — |
| 7 | W | 115 | 0.96 | `srflx`–`srflx` UDP 17 / 18 ms |
| 8 | R | 297 | 0.94 | — |
| 9 | P | 323 | **69.6** | **neither stage's link opened** (relay fallback) |

| Check | Result |
|---|---|
| Request 1 and 2 texts identical in all 9 sessions | pass |
| Request 2 at most 8 tokens | pass (8 in every session) |
| No `disagreed` line; no IPv4 address in route records | pass |
| W at least 25% below R | pass: 115 vs 287 ms, −60% |
| P at most R | pass: 284 vs 287 ms, −1% |
| Every W and P link open, P `relay`–`relay` | **fail:** session 9 |
| First token within 5 s in every session | **fail:** session 9, 69.6 s |

The 8-token request: R 179–303 ms per token, W 107–119 ms, P 173–204 ms. The whole first request took
15.0–16.6 s on R, 6.7–7.6 s on W, 15.3–15.6 s on P sessions that linked, and 225.5 s on session 9.

Diagnostic after the matrix, outside the gate: three P sessions, all `relay`–`relay` 59–64 ms, open at
`go`, first token 1.1–1.7 s, 274–285 ms per token.

## What it taught

- **The fixes work in the common case.** Every direct link and 7 of 8 private sessions (counting the
  rehearsal and the diagnostic) linked within a second of `go`; experiment 98's 30 s waits are gone.
- **With close stages, the user path gets experiment 97's gain.** Direct links cut time per token 60%
  (287 → 115 ms), the same as 97's −61% on the experiment path; with experiment 98's distant stages it
  was −11%. Direct links pay off when contributors are near each other, and much less when the
  coordinator is about as close to each as they are to each other.
- **Private links through a TURN server next to the coordinator cost about what the relay does**
  (−1%): every handoff still goes to Amsterdam and back. Their value is privacy, not speed; a TURN
  server near the contributors would be faster.
- **A rare failure remains.** In session 9 neither link opened, stage A took 70 s to finish connecting
  (its 30 s wait starts only after its connection set-up returns, which waited on retried calls), and the
  turn then had stalls of many seconds although it ran through the relay. The coordinator's access log was
  off and verbose TURN logging was on only for the diagnostic, so the cause is not known. Two things to
  fix whatever it is: the 30 s limit must cover the whole set-up, not start after it; and a stage whose
  link is not open must not hold messages for it.

## Limits

- One pair of hosts per geometry, three sessions per arm.
- Datacenter containers: no home NAT; W never needed TURN.
- Session 9's failure is unexplained (see above); its frequency, 1 in 8 private sessions here, is a rough
  estimate from few sessions. No direct (W) session failed in this run, 4 of 4 including the rehearsal.

## Reproduce

As experiment 98's Reproduce, with two GPUs in one country on different machines (check that their
addresses differ), pools named do44a, do44b and do44coord, and `runs/experiments/DO-44/launch_remote.py`
and `gate_remote.py`:

```bash
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do44a --provider vast \
  --location PL --server any-8gb --workers 0 --minutes 115 --max-hourly 0.3 --no-collect
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do44b --provider vast \
  --location PL --server any-8gb --workers 0 --minutes 115 --max-hourly 0.3 --no-collect
python -m ygaraty.experiments.pods up do44coord --provider digitalocean --location ams3 --server s-4vcpu-8gb \
  --workers 0 --minutes 110 --max-hourly 0.3 --no-collect
# set-up, panel-check main --arms R,W,P,P,W,R,W,R,P --max1 128 --max2 8 --max3 32, launch_remote.py main
# --hosts do44a:server do44b:server, gate_remote.py OUT, then pods down --no-fetch: as in experiment 98
```
