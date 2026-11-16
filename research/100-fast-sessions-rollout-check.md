# Fast (direct) rollout check did not complete: the relay baseline hung on a Poland–Germany pair, while Fast answered 7 of 8 sessions

| | |
|---|---|
| **Conclusion** | **Did not complete:** the relay baseline hung on this host pair; Fast answered 7 of 8 sessions (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-45 |
| **Date** | 2026-11-09 |
| **Path** | As [experiment 99](099-user-sessions-webrtc-close-stages.md) (user-panel path, S5-nano two-stage bundle, coordinator in DigitalOcean `ams3`), with the link deadlines and health checks of e9cb1aa and the acknowledged resend of fe0c209. Stage A a Vast.ai RTX 3060 in Poland; stage B an RTX 3080 Ti in Germany (protocol amendment before any run: no second Polish machine was available). Arms R relay and W Fast (direct) |
| **Result** | **The gated matrix never ran.** In the rehearsals and five check runs (17 sessions), relay sessions answered all their requests in only 3 of 9, and every hung relay request had the German host as the last stage; with the stages swapped, both relay sessions answered. Fast answered every request in 7 of 8 sessions, links open `srflx`–`srflx` (33–37 ms) within 0.6–4.4 s of being made, first token about 1 s, 115–165 ms per token against 192–254 ms on the relay sessions that answered (not a gated comparison). The one Fast failure was the very first session, while stage A was still downloading its weights. Cancel and replacement were not reached on rented machines. Three test-harness bugs were found (see What it taught); the stop command is now `remote.sh qwen35-browser-stop`. Cost about $0.36 of the $1 budget |
| **Evidence** | Measured: [evidence/browser/fast-sessions-rollout-check-2026-11-09.json](evidence/browser/fast-sessions-rollout-check-2026-11-09.json) (every session's outcome, links, times); raw results and session logs in `runs/experiments/DO-45/` (not in git). Code fe0c209, 29f65bc; launcher `runs/experiments/DO-45/launch_remote.py` |

## Question

Experiment 99 measured Fast at −60% but failed on link reliability (one session linked after 70 s and
stalled). With every link now held to 30 s from `go`, dropped when it stalls or disconnects, and nothing
lost when it dies, is Fast ready to be offered: same answers, every link open at once, still much faster
than the relay, and a cancelled turn and a replaced contributor handled?

## Prediction

Every W link opens within a second or two of `go`; W about −60% against R as in experiment 99; cancel and
replacement keep the answer on both arms, and W's new stage B relinks at once.

## Gate

Frozen 2026-11-09 13:58 CET, before any run.

- Two Vast.ai GPU hosts on different machines in one country; the coordinator as in experiment 99.
  Sessions in the order R W W R R W, each with a 23-token prompt (`max_tokens` 128), the conversation
  continued (`max_tokens` 8), a new prompt (`max_tokens` 32, not gated), then `--cancel` (a streamed copy of
  request 1 dropped after its first chunk, then request 1 again) and `--replace` (stage B's browser leaves,
  a new one takes the stage, request 1 again).
- **Correctness:** requests 1 and 2 identical across all six sessions; request 2 at most 8 tokens; the
  requests after cancel and after replacement identical to request 1 in every session; no `disagreed`
  line; no IPv4 address in any route record.
- **Links:** in every W session both stages report their link open at `go`, the first request's first
  token comes within 5 s, and after the replacement the new stage B reports its link open.
- **Works** if W's median time per token (first request, median of its 3 sessions) is at least 25% below
  R's.
- Any failed line fails the gate. Computed by `runs/experiments/DO-45/gate_remote.py`.
- If it passes, `allow_webrtc` is set on the S5-nano plans in `demo_inference.json`; production's
  `YGARATY_WEBRTC_ENABLED` stays the owner's decision.

## Protocol

As experiment 99 (`remote.sh panel-coordinator-setup`, `turn-serve`, `panel-coordinator-serve`, then
`panel-check main --arms R,W,W,R,R,W --max1 128 --max2 8 --max3 32 --cancel --replace`, with
`runs/experiments/DO-45/launch_remote.py` opening each link, the replacement's too, on its stage's host),
from fe0c209 or later, nothing copied by hand.

## Log

- 2026-11-09 13:58 CET: added; gate frozen. Budget **$1**, teardown deadline **15:58 CET** (2 h),
  approved by the user.
- 14:04: pools do45a (Poland, RTX 3060) and do45coord (`ams3`). Three tries for a second Polish GPU
  (do45b, do45c, and the named RTX 3080 / RTX 4060 Ti offers) gave the same machine as do45a or no match;
  do45b and do45c deleted at once. **Protocol amendment, before any run:** the second GPU is do45f in
  Germany (RTX 3080 Ti), on another machine; the pass criteria are unchanged. Watchdogs 106–115 min.
- 14:07: setup done. Rehearsal: panel-check rehearsal --arms W --max1 16 --max2 4 --max3 8 --cancel --replace.
- 14:26: rehearsals: the first session of each check run got no token (W in the first, R in the second), the next one worked. Diagnostic run R,R,W,W.
- 14:33: diagnostic: browser files of 3e80cd9 (experiment 99) served temporarily; four R sessions.
- 14:45: the old browser code hung the same way (session 2 of 4): not a regression. Launcher now waits for every Chrome on the shared profile to exit before starting the next; current browser files restored (hashes checked). Check run R,R,W,W.
- 14:58: check run 'fix' (R,R,W,W): both R sessions hung again, both W fine. Probe: a German-host Chrome started but its page never joined; 14–15 leaked xvfb-run X servers on each host (stopping the wrapper left them). Launcher now stops them too; hosts cleaned.
- 15:06: check run 'fix2' (R,R,W,W, X servers cleaned): R hung again (3 of 4 relay requests in two sessions), W 2 of 2 clean. Last test: stages swapped between hosts, two R sessions.
- 15:12: swapped stages: both relay sessions answered (one at about 17 s per request). Teardown.
- 15:13: all pools deleted; `pods ls`: 0 machines.

## Results

| Run | Session | Arm | Requests answered | ms per token (request 1) | Links |
|---|---|---|---|---|---|
| rehearsal | 1 | W | 0 of 1 | — | open `srflx`–`srflx` (stage A still downloading until go) |
| rehR | 1 | R | 0 of 1 | — | — |
| rehR | 2 | W | 3 of 3 | 119 | open, 1.2–1.3 s |
| diag | 1 | R | 3 of 3 | 197 | — |
| diag | 2 | R | 0 of 1 | — | — |
| diag | 3–4 | W | 3 of 3 each | 120, 115 | open, 0.7–4.0 s |
| fix | 1 | R | 1 of 2 | 192 | — |
| fix | 2 | R | 0 of 1 | — | — |
| fix | 3–4 | W | 3 of 3 each | 165, 117 | open, 0.6–2.9 s |
| fix2 | 1 | R | 2 of 3 | 202 | — |
| fix2 | 2 | R | 0 of 1 | — | — |
| fix2 | 3–4 | W | 3 of 3 each | 118, 122 | open, 0.6–2.8 s |
| swap (Germany as stage A) | 1–2 | R | 3 of 3 each | 254, 204 | — |

A diagnostic run with experiment 99's browser files hung the same way (session 2 of 4), so the hangs do not
come from the browser changes since then. The gate was not evaluated.

## What it taught

- **The relay depends on each contributor's HTTP path to the coordinator; Fast mostly does not.** With the
  German host as the last stage, relay requests stalled until the 120 s turn timeout in most sessions, while
  Fast sessions on the same hosts answered. The cause on that path (loss, a middlebox, the long-poll through
  the coordinator's TLS proxy) was not found; the coordinator's access log was off.
- **The fixes of e9cb1aa and fe0c209 behaved in the sessions that ran:** every Fast link opened within 4.4 s of
  being made, the first token came in about 1 s, and no session waited 30 s.
- **Three harness bugs:** the launcher's stop command matched its own text and killed its own shell (a
  self-matching pattern); stopping the browser wrapper left its `xvfb-run` X server running (14–15 per host
  after about 20 sessions, when a new Chrome's page stopped joining); and a new Chrome could start on the
  shared profile while the previous one was still exiting. `remote.sh qwen35-browser-stop` now stops
  wrappers, waits for Chrome on the profile to exit, stops leaked X servers and clears the profile lock.
- **Picking hosts:** the cheapest Polish offers were all on one machine; a run that needs two machines in one
  country should check the offers first or exclude the first machine.

## Limits

- The gate was not evaluated: no relay baseline, and cancel and replacement never ran remotely.
- One host pair; the German host's network was the problem, so another pair may behave like experiment 99.
- The Fast numbers above come from check runs with short requests (16 tokens), not the gated 128-token run.

## Reproduce

As experiment 99's Reproduce, with `panel-check main --arms R,W,W,R,R,W --max1 128 --max2 8 --max3 32 --cancel
--replace`, `runs/experiments/DO-45/launch_remote.py` (or `remote.sh qwen35-browser-stop` before each
`qwen35-browser`) and `runs/experiments/DO-45/gate_remote.py`.
