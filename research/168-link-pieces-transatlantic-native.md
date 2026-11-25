# On native contributors, WS-26's piece schedule makes no difference on a real transatlantic TURN link: both arms take about 328 ms per token

| | |
|---|---|
| **Conclusion** | Rejected: piece schedule 328.8 vs 328.2 ms per token over a 130 ms TURN link, native contributors (measured) |
| **Status** | executed |
| **ID** | DO-54 (report 168) |
| **Category** | browser |
| **Where** | remote (DigitalOcean coordinator + coturn in fra1; Vast.ai RTX 4060 Ti 16 GB in Norway and in Ontario) |
| **Priority** | high |
| **Date** | Added 2026-11-11; first attempt 2026-11-11 (did not complete); rerun 2026-11-24 |
| **Path** | Phi-4 Q4_K two stages (0,20,40), route `turn`, native contributors (Node + Dawn + node-datachannel). Arm B: `linkPieces=0` (bodies in 64 KB sends, as bd3354a); arm F: WS-26's schedule (today's `qwen35_peer.mjs`, identical to 4764565's) |
| **Result** | All six sessions valid (3 turns of 256 tokens; stage link relay–relay through TURN, 128–137 ms). Session medians, ms per token: B1 328.0, F2 323.5, F3 328.8, B4 328.2, B5 332.6, F6 330.1. B 328.2 against F 328.8: ratio 1.002 (Works needed ≤ 0.8 and every F below every B). The 2026-11-11 attempt did not complete (no session reached a turn; logs lost) |
| **Evidence** | measured-remote: `runs/experiments/DO-54/remote-20261124/` (per session `X.json` with per-turn timing and text, `X.serve.log`, contributor logs); the harness's first try in `remote-20261124-harness1/` |

## Question

[WS-26](153-link-pieces-local.md) removed the extra link round trip per decode step on a local link with a constant artificial delay (583 → 341 ms per token at 243 ms). Does the same hold on a real transatlantic, TURN-relayed link with real loss and jitter ([DO-50](DO-50.md) round 2's path, where tokens came too slowly)?

## Prediction

Per token ≈ compute + 2 × RTT (B) and compute + 1 × RTT (F); at DO-50's 238 ms, about 240 ms per token less with F.

## Budget

User: "run 1, 2 and 3 within budget" after "any budget below 10 is approved" (2026-11-11). Cap for this experiment: **$5**; teardown: pool watchdogs at **240 min** after `pods up`; estimate about $0.66/h × 3 h ≈ $2.

## Gate (frozen 2026-11-11, before renting)

- Sessions: one per arm, in the order B F F B B F, each on the same three machines and the same Phi-4 bundle (`--capacity 4096 --generate 256`). The arm is the coordinator's `ygaraty/webapp/static/qwen35_peer.mjs` (served `no-store`; swapped, with its hash checked, before each session's coordinator restart); a fresh browser per session. Each session: the chat driver (`chat_cycles.py`, HTTP chat, `--cycles 1 --turns 3 --words 1500 --token-cap 256 --timeout 900`).
- Measure: per turn, ms per token from the coordinator's turn line (`N tokens, first after X s, Y ms per token`); per session, the median over its 3 turns; the stage link's candidate types and RTT from the route log.
- Valid session: 3 turns completed with 256 tokens each; the stage link open and through TURN (a `relay` candidate on at least one side); its RTT ≥ 100 ms. An invalid session is reported and not used.
- **Works** if the median over F sessions ≤ 0.8 × the median over B sessions and every F session is below every B session. Otherwise **Rejected**. Fewer than 2 valid sessions per arm: **Inconclusive**.
- Correctness is not re-tested here (WS-26 showed bitwise equality); the reply texts of B and F sessions are compared descriptively (the driver now hashes the reply only, WS-27).

## Gate, rerun (frozen 2026-11-24, before renting)

The 2026-11-11 gate's criteria, unchanged: sessions B F F B B F on the same machines and bundle; per turn, ms per token from the coordinator's `chat turn` line; per session, the median of its 3 turns. A session is valid with 3 turns of 256 tokens each and the stage link open through TURN (a `relay` candidate on at least one side) with round trip ≥ 100 ms. **Works** if F's median ≤ 0.8 × B's median and every F session is below every B session; otherwise **Rejected**; fewer than 2 valid sessions per arm: **Inconclusive**.

Deviations from 2026-11-11, decided before renting:
- Contributors are native (`native_contributor.mjs`, Node + Dawn, `--webrtc-module` with node-datachannel 0.29.0), not Chromium. libdatachannel also runs usrsctp, the SCTP stack whose first-flight limit WS-26 worked around; a Chromium result is not re-measured here.
- The arm is the link parameter `linkPieces=0` (B: bodies in 64 KB sends, as bd3354a) on today's `qwen35_peer.mjs` (F: the WS-26 schedule, identical to 4764565's file), instead of swapping the coordinator's static file.
- Route `turn` (TURN only), so the link always goes through coturn on the coordinator.
- Bundle: `ygaraty model prepare` of `microsoft/phi-4-gguf` `phi-4-Q4_K.gguf` at 6edc2ef (sha256 5652b9be…), config and tokenizer from `microsoft/phi-4` at 2db69c1c, splits 0,20,40, context 4,096, 256 generated tokens.
- Driver: `runs/experiments/DO-97/tools/ask.py` with three ~1,550-word prompts (DO-95's fact list with three long-answer instructions, `DO-54/tools/prompts.json`), `max_tokens` 256, temperature 0; each request is one turn. Each session's logs are copied to the workstation when it ends, and the driver stops at the first session without three turns.

## Protocol

- Pools: `do54c` DigitalOcean s-8vcpu-16gb fra1; `do54a` Vast.ai RTX 3090 in Europe; `do54b` Vast.ai RTX 3090 in the US or Canada; `--minutes 240`. Code: a clone of `main` at b1bcc5b (arm F); arm B's `qwen35_peer.mjs` from bd3354a, hand-copied with its hash. Harness: DO-50 round 2's (`runs/pods-data/do54/`: `do54.sh` from `do50r2.sh` with a 256-token bundle; `serve_r2.py`, `launch_r2.py` unchanged).
- `pods down` as soon as the last session ends; `pods ls` must show 0.

## Log

- 2026-11-11: Added at user request (item 3 of "what is next?"); gate frozen before renting.
- 2026-11-11 ~13:20 CET: rented. `do54c` DigitalOcean s-8vcpu-16gb fra1 ($0.143/h); `do54a` Vast.ai RTX 3090 France ($0.209/h); `do54b` Vast.ai RTX 3090 United States ($0.256/h); watchdogs 239 min; $0.61/h in all, at most about $2.6. Hand-copied to the droplet: `do54.sh` 5c74ccb9…, `serve_r2.py` 56bffa15…, `qwen35_peer.B.mjs` 6b3ba5f0… (bd3354a), `qwen35_peer.F.mjs` b3e15472… (4764565), `chat_cycles.py` 3cbd11a5… (as pushed in 4764565). Setup started on all three (the first start on the GPU hosts did not detach and never ran; restarted with `setsid -f`).
- 13:30 UTC droplet setup done (Phi-4 fetched, `phi-4/c256/2-workers` prepared); GPU hosts' setup and WebGPU probes passed (US host at 13:41 UTC). 14:43 CET: matrix started (`runs/pods-data/do54/matrix.sh "B1 F1 F2 B2 B3 F3"`, one `do54.sh runs` per session with its `ARM`, the workstation's `launch_r2.py` opening the contributor links).
- Two harness faults before any session ran (no data lost, gate unchanged): (1) the run spec, passed through ssh and `bash -c` quoting, reached `serve_r2.py` as invalid JSON (B1 exited at once and the launcher waited); the spec now lives in `/root/do54-spec.json` (138a2b51…) and the matrix releases the launcher when a session exits without `done`. (2) `do54.sh` defaults `MEMORY_GB=12`, which the panel's plan constraint (2–8) refuses; round 2 had set 8 by hand; now passed as `MEMORY_GB=8`. ~15:05 CET: matrix restarted; B1's contributor links issued.

- 15:05–17:41 CET: B1 15:05–15:39, F1 15:39–16:20 (launcher: ssh to `do54b` timed out during banner exchange), F2 16:20–17:01, B2 17:01–17:41: each launcher opened both stage browsers within 30 s and returned `done` about 40 min later, but the serve log's last line was `session preparing` and the coordinator log held no `chat turn` line, so no session reached a turn (cause not found: the logs were on the droplet). 17:41: B3 could not start (the droplet was gone). 18:14–18:17: the watchdogs deleted `do54c`, `do54a`, `do54b`; their fetch saved empty directories. 20:14: matrix stopped; `pods ls` 0 machines. **Did not complete.** Lesson: copy each session's logs to the workstation as soon as it ends, and stop the matrix at the first session without turns instead of letting it run on.
- 2026-11-24: Queued at the user's request ("queue do62, do54, do47, do97 and m8"). Order M-8, DO-97, DO-54, DO-62, DO-47; each remote run up to $3 and 150 minutes of machines, gate frozen before renting. Queued third: the harness is pre-refactor and must be rebuilt first.

## Limits

One pair of hosts and one path; Chromium only; S5-nano's 5 KB and Phi-4's 10 KB hidden states both fit one flight with the schedule, larger models not tested.
- 2026-11-24: **Harness rebuilt** for native contributors (`runs/experiments/DO-54/tools/`: `coord.sh` for Phi-4, `drive.sh`, `prompts.json`; DO-97's `contrib.sh` now takes link parameters). `qwen35_peer.mjs` reads `linkPieces=0` (test `tests/webgpu/browser/link_pieces.test.mjs`). Local check on the tiny Inkling bundle, three native contributors, direct links: `linkPieces=0` gives the relay's tokens. Gate re-frozen above with its deviations; budget $3 and 150 minutes of machines (queue).
- 2026-11-24: **Rerun.** Coordinator DigitalOcean `s-8vcpu-16gb` in fra1 (`do54c2`, $0.143/h): Phi-4 GGUF in 2 minutes (sha256 5652b9be… OK), bundle (sha256 31c96af4…, 2 stages, Phi-4's `<|im_sep|>` chat format from its template) in 2 minutes, coturn on 3478. Two RTX A4000s chosen first were taken before renting; contributors became RTX 4060 Ti 16 GB in Norway (`do54eu`, $0.171/h, 31 ms to the coordinator) and in Ontario (`do54us`, $0.139/h, 100 ms). All deleted by their recorded IDs; about **$0.51** of $3.
- 2026-11-24: **Harness fix during the run, before any gated data.** The first session (`remote-20261124-harness1/`) showed two faults: `ask.py` times names starting with `t` as 1- and 33-token requests, so the `turnN` prompts never ran 256 tokens; and today's chat API path writes no `chat turn` line (only the old panel chat did). Each turn is now timed from the streamed chat API (`tools/turns.py`: first token time, and ms per token = (last chunk − first chunk) / (tokens − 1), the old line's definition; tokens from the final usage). The six sessions were then rerun from the start.
- 2026-11-24: **Sessions** (B F F B B F, 18:20–19:11 UTC; per turn, ms per token):

  | Session | Turn 1 | Turn 2 | Turn 3 | Median | First token, turns 1–3 (s) | Link round trip |
  |---|---:|---:|---:|---:|---|---:|
  | B1 | 349.2 | 324.5 | 328.0 | 328.0 | 13.4, 12.0, 11.8 | 130 ms |
  | F2 | 323.5 | 325.9 | 322.7 | 323.5 | 22.5, 11.9, 12.1 | 137 ms |
  | F3 | 340.5 | 325.4 | 328.8 | 328.8 | 15.6, 11.9, 12.0 | 131 ms |
  | B4 | 342.3 | 327.2 | 328.2 | 328.2 | 16.1, 12.0, 11.9 | 130 ms |
  | B5 | 334.0 | 332.6 | 328.9 | 332.6 | 16.0, 12.0, 11.9 | 129 ms |
  | F6 | 345.4 | 330.1 | 329.4 | 330.1 | 15.3, 12.1, 12.1 | 137 ms |

  Reply texts (descriptive): identical in all sessions for turns 2 and 3, and for turn 1 in five of six; F2's first reply differs.
- 2026-11-24: **Gate**: Rejected (F/B 1.002; F2 is the only F session below every B).

## Conclusion

On native contributors the piece schedule changes nothing: both arms take about 328 ms per token over a 130 ms TURN link. WS-26's gain came from Chromium's SCTP sending only about four packets per `send()` before waiting for an acknowledgement, so a whole 10 KB body cost an extra round trip. Here arm B already completes each hand-off in one flight. libdatachannel's usrsctp settings evidently differ from Chromium's, and the per-token time is about one link round trip plus the stages' work in both arms. The question for browsers stays open: this rerun did not use Chromium, and the 2026-11-11 Chromium attempt never reached a turn.

## Limits

- Native contributors only; no Chromium session.
- One path (Norway–Frankfurt–Ontario) and two RTX 4060 Tis instead of the planned RTX 3090s.
- Turn timing from the client's receive times on the coordinator machine, not the coordinator's internal record.

## Reproduce

Coordinator (Ubuntu, public IP, UDP open): `runs/experiments/DO-54/tools/coord.sh setup`, `ygaraty init --directory /root/r/keys`, `coord.sh turn SECRET`; kit with `turns.py` and `prompts.json`. Contributors: DO-97's `contrib.sh setup` and `contrib.sh proxy COORD:8000`. From the workstation: `C=... G1=... G2=... CK=... OK=... bash runs/experiments/DO-54/tools/drive.sh OUT "B F F B B F"`.
