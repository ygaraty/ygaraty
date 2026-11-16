# Fast (direct) passes the rollout gate: same answers, links open at go, 63% less time per token than the relay

| | |
|---|---|
| **Conclusion** | **Works:** Fast (direct) passes every rollout gate line, 63% less time per token than relay (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-46 |
| **Date** | 2026-11-09 |
| **Path** | As [experiment 100](100-fast-sessions-rollout-check.md) (DO-45): user-panel path, S5-nano Q4_K_M two-stage bundle (0,16,32), coordinator with coturn in DigitalOcean `ams3`; stage A a Vast.ai RTX 3060 (Poland, machine 20698), stage B an RTX 3060 on another machine (Poland, 48515, another public IP in the same /24). Code e0f63b4 (41eb181's bounded mailboxes and admission checks plus a template change). Arms R relay and W Fast (direct), order R W W R R W, each with cancel and replacement |
| **Result** | **Gate passed, all eleven lines.** Every session answered every request; requests 1 and 2 identical in all six (52 and 8 tokens); after a cancelled turn and after a replaced stage B, request 1 unchanged in every session; no disagreement, no IPv4 in route records. W: both links open `srflx`–`srflx` at 1–2 ms within 0.5–1.6 s, the replacement relinked in 0.6 s. First token 0.64–0.97 s. Median time per token (request 1) 110 ms on W against 298 ms on R (−63%, gate −25%). Cost about $0.09 of the $1 budget. `allow_webrtc` is now set on the S5-nano plans in `demo_inference.json` |
| **Evidence** | Measured: [evidence/browser/fast-sessions-rollout-rerun-2026-11-09.json](evidence/browser/fast-sessions-rollout-rerun-2026-11-09.json) (gate lines, per-session tokens, times, links, cancel and replacement); raw results, session logs and check logs in `runs/experiments/DO-46/` (not in git); local smoke run `runs/experiments/DO-46/local-20261109-2016`. Gate script `runs/experiments/DO-46/gate_remote.py`, launcher `launch_remote.py` |

## Question

Experiment 100 never evaluated its gate: the relay baseline hung on a Poland–Germany host pair. On a host
pair whose relay works, is Fast (direct) ready to be offered: same answers, every link open at once, at least
25% faster than the relay, and a cancelled turn and a replaced contributor handled?

## Prediction

As experiment 100's: W links open within a few seconds of `go`; W about −40 to −60% against R; cancel and
replacement keep the answer on both arms.

## Gate

Frozen 2026-11-09 20:15 CET, before any run. Experiment 100's gate unchanged, with a host-pair rule:

- **Hosts:** two Vast.ai GPU hosts on different machines (different machine IDs), in one country or
  neighbouring ones; the coordinator as in experiment 100. Before the gated run, a rehearsal
  (`--arms R,W`, short requests, cancel and replace) must answer every request in **both** arms; if the
  relay arm hangs, the pair is replaced (at most once) and the rehearsal repeated. This is host selection,
  decided before the gated matrix, and is recorded in the Log.
- Sessions in the order R W W R R W, each with a 23-token prompt (`max_tokens` 128), the conversation
  continued (`max_tokens` 8), a new prompt (`max_tokens` 32, not gated), then `--cancel` and `--replace`.
- **Correctness:** requests 1 and 2 identical across all six sessions; request 2 at most 8 tokens; the
  requests after cancel and after replacement identical to request 1 in every session; no `disagreed`
  line; no IPv4 address in any route record.
- **Links:** in every W session both stages report their link open at `go`, the first request's first token
  comes within 5 s, and after the replacement the new stage B reports its link open.
- **Works** if W's median time per token (first request, median of its 3 sessions) is at least 25% below R's.
- Any failed line fails the gate. Computed by `runs/experiments/DO-46/gate_remote.py` (experiment 100's script).
- If it passes, `allow_webrtc` is set on the S5-nano plans in `demo_inference.json`; production's
  `YGARATY_WEBRTC_ENABLED` stays the owner's decision.

## Protocol

As experiment 100 (`remote.sh panel-coordinator-setup`, `turn-serve`, `panel-coordinator-serve`, then
`panel-check main --arms R,W,W,R,R,W --max1 128 --max2 8 --max3 32 --cancel --replace` with
`runs/experiments/DO-46/launch_remote.py`), nothing copied by hand. A local smoke run of the same flags on
this workstation first (free), because 27fc315 changed the coordinator's mailboxes and token reports.

## Log

- 2026-11-09 20:15 CET: added; gate frozen. Budget **$1**, teardown deadline **22:20 CET** (2 h), approved
  by the user.
- 20:17: local smoke run (`runs/experiments/DO-46/local-20261109-2016`, arms R,W, cancel and replace): both arms answered every request, cancel and replacement kept the answer, W relinked the new stage B at once. Renting: do46a (Poland, RTX 3060), do46b (Poland, RTX A4000, another machine), do46coord (`ams3`); watchdogs end at 22:15.
- 20:22: do46b (RTX A4000, machine 18110) had the same public IP as do46a (machine 20698): deleted at once, unused. Second host do46c: RTX 3060, machine 48515, another public IP in the same /24 (likely the same site), so links will be shorter than experiment 99's 14–18 ms; recorded, pass criteria unchanged.
- 20:25: setup done (all three cloned e0f63b4). Rehearsal (host selection): panel-check rehearsal --arms R,W --max1 16 --max2 4 --max3 8 --cancel --replace.
- 20:30: rehearsal passed on both arms (R 3 of 3 plus cancel and replacement, W the same, links `srflx`–`srflx` 1–2 ms, opened 0.5–1.5 s); machines run e0f63b4 (41eb181 plus a template change). Host pair accepted. Main matrix: panel-check main --arms R,W,W,R,R,W --max1 128 --max2 8 --max3 32 --cancel --replace.
- 20:39: main matrix finished; session logs and results copied (contributor links, which hold keys, left
  behind). Gate computed by `gate_remote.py`: passed.
- 20:41: all pools deleted (`down --no-fetch`: the hosts have no `runs/`); `pods ls`: 0 machines. Cost about
  $0.09.


## Results

| Session | Arm | Tokens (requests 1–3) | ms per token, request 1 | First token | Links at `go` | After replacement | Cancel / replace same |
|---|---|---|---|---|---|---|---|
| 0 | R | 52, 8, 32 | 293 | 0.97 s | — | — | yes / yes |
| 1 | W | 52, 8, 32 | 113 | 0.78 s | open 0.6 s, 1.6 s (2 ms) | open 0.6 s | yes / yes |
| 2 | W | 52, 8, 32 | 109 | 0.73 s | open 1.0 s, 0.5 s (1–2 ms) | open 0.6 s | yes / yes |
| 3 | R | 52, 8, 32 | 304 | 0.66 s | — | — | yes / yes |
| 4 | R | 52, 8, 32 | 298 | 0.77 s | — | — | yes / yes |
| 5 | W | 52, 8, 32 | 110 | 0.64 s | open 0.6 s, 1.6 s (2 ms) | open 0.6 s | yes / yes |

Medians of request 1: R 298 ms, W 110 ms per token (−63%). Request 1 stopped at 52 tokens (the model's end of
answer) in every session: "The sky appears blue because sunlight, which is white light containing all colors,
gets scattered by air molecules…". A replaced stage B was serving again 18–22 s after leaving on both arms.

The host-selection rehearsal (R, W; 16/4/8 tokens, cancel and replace) answered every request on both arms
before the gated matrix; the local smoke run on the workstation did the same.

## What it taught

- **Experiment 100's failure was the host pair, not the code.** On a pair with a working relay, the same
  protocol (plus 27fc315's bounded mailboxes and token-report limits) passed every line on the first run.
- **Fast's advantage held at −63%**, as experiment 99's −60% on a pair 14–18 ms apart; here the stages were
  1–2 ms apart, so the gain comes from leaving the coordinator out of the per-token path (about 30 ms each way
  from Poland to Amsterdam), not from the short link.
- **Picking Vast.ai hosts:** two machine IDs can share one public IP; check the addresses, not just the IDs.

## Limits

- One host pair, at one site (1–2 ms apart): home contributors farther apart gain less per hop but still skip
  the coordinator round trip.
- Only the 2-stage S5-nano plan ran remotely; `allow_webrtc` is also set on the 3- and 4-contributor S5-nano
  plans, which use the same link code (checked locally in experiment 102 for other models).
- The private (TURN-only) route was not part of this gate; it stays disabled in the form.
- Production's `YGARATY_WEBRTC_ENABLED` is the owner's decision.

## Reproduce

As experiment 100's Reproduce: `remote.sh panel-coordinator-setup`, `turn-serve`, `panel-coordinator-serve` on
the coordinator (`YGARATY_QWEN_MODEL=s5-nano`), `qwen35-contributor-setup` on both GPU hosts, then

```bash
pods ssh <coord> server "bash /root/remote.sh panel-check main --arms R,W,W,R,R,W --max1 128 --max2 8 --max3 32 --cancel --replace"
python runs/experiments/DO-46/launch_remote.py main --hosts <A>:server <B>:server --coord-pool <coord>
# copy panel-check-main/results.json and serving/<session>/session.log as <session>.log, then
python runs/experiments/DO-46/gate_remote.py OUT
```
