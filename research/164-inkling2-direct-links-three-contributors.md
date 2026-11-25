# Direct links between Inkling 2's three contributors keep the tokens but do not halve time per token in one region: the GPUs, not the hops, set the pace

| | |
|---|---|
| **Conclusion** | Failed gate: direct links 298 vs 285 ms per token over the relay, same tokens (measured, 3 Vast.ai GPUs, one region) |
| **Status** | executed |
| **ID** | DO-97 (report 164) |
| **Category** | browser |
| **Where** | workstation (native WebRTC, local check); remote (DigitalOcean coordinator + coturn in Toronto, three Vast.ai GPUs in Quebec) |
| **Priority** | med |
| **Date** | Added 2026-11-22; started 2026-11-24; finished 2026-11-24 |
| **Path** | Report 156's arm B (layers 0–5 \| 6–17 \| 18–23, `f32` wire) with `--route webrtc`: native contributors (Node + Dawn) with node-datachannel's `RTCPeerConnection` (`--webrtc-module`), ICE servers from coturn on the coordinator. Control: `--route relay` on the same machines, blocks R D D R |
| **Result** | Item 1 passes: chat, code and long give report 156's 64 greedy tokens in all 4 blocks. Item 2 **fails**: decode at 1k positions, D median 298.1 ms per token (317.5, 278.8) against R 284.5 (280.8, 288.2), ratio 1.05 (bound ≤ 0.5; the ≤ 376 ms bound passes). Item 3 passes: 6 of 6 directed links open in both D blocks, none dropped; the coordinator saw 57 receive polls per D block against about 816 with the relay. Not gated: 64-token chat and code requests were about 10% faster direct (15.8–17.0 vs 17.4–18.6 s); first token at 1k 14.8–15.0 vs 15.6–15.9 s |
| **Evidence** | measured-remote: `runs/experiments/DO-97/remote-20261124b/` (`tools/gate.py` output below; per block `N-X.json`, `N-X.serve.log`, contributor logs); first attempt with broken TURN in `remote-20261124/`; local check in `omarchy-20261124-local/` |

## Question

Report 156's arm B decoded at about 750–790 ms per token through the relay, against 160 ms on one A100. Each token makes three relayed hops (A→B, B→C, C→A's next token), each an HTTP send plus a poll through the coordinator. Two of those hops crossed Delaware–Utah.

With direct links (experiment 103: 63% less time per token than the relay for Qwen), how close does a three-contributor Inkling session get to one GPU? What do the 18–37 KB `f32` hand-offs per token cost on real links?

## Prediction

With all three contributors in one region (round trip ≤ 20 ms), decode is about 160–250 ms per token: three link hops plus the stages' compute. Across regions it is about one round trip per hop more. The tokens are unchanged, since the route changes where bytes go, not what is computed.

## Gate (frozen 2026-11-24, before renting)

Same three machines, same coordinator, same code, both arms in one session: **R** (`--route relay`) and **D** (`--route webrtc`, native contributors with `--webrtc-module`; ICE servers from coturn on the coordinator). Blocks R D D R, so each arm runs twice.

1. **Same tokens:** in every block, the 64 greedy tokens for chat (27-token prompt), code (30) and long (2,761) are identical, as decoded text, to report 156's answers (`runs/experiments/DO-95/vast-do95b-20261122/client-partial.json`, which equal arm A's).
2. **Faster:** D's decode ms per token at 1k positions (the `t1k` prompt: 33-token minus 1-token request, as report 156) is at most half of report 156's relay arm B (752 ms, so ≤ 376 ms) **and** at most half of R on the same machines (median of R's blocks against median of D's).
3. **Links:** in every D block all three pairs open at go, and none drops or falls back to the coordinator relay during the run (contributors' per-pair route reports). Reported, not gated: candidate types (host, srflx, relay through TURN) and round trips per pair, prefill time at 1k, bytes per hop.

The 8k point is not part of the gate (the coordinator's 120 s request deadline; report 156 needed 1- and 8-token requests). If time is left it is reported for D only.

## Protocol

1. Prerequisite (workstation, done 2026-11-24): `native_contributor.mjs --webrtc-module PATH` installs node-datachannel 0.29.0's `RTCPeerConnection` polyfill; without it the contributor stays relay-only. Checked on the tiny-GGUF 3-contributor session locally: same tokens as the relay.
2. Three rented GPUs with at least 12 GB in one region, chosen by measured bandwidth (report 156's offer query), plus a TURN server if needed. Budget and teardown deadline from the user first.
3. Optionally a second run across regions to price the hops.

Related: report 156 (DO-95), experiment 103 (Fast/direct links), WS-51 (relay backpressure), DO-96 (faster decode, which makes the link share of each token larger).

## Log

- 2026-11-22: Added after report 156.
- 2026-11-24: Queued at the user's request ("queue do62, do54, do47, do97 and m8"). Order M-8, DO-97, DO-54, DO-62, DO-47; each remote run up to $3 and 150 minutes of machines, gate frozen before renting. Queued second: native contributors need WebRTC first; then 3 rented GPUs.
- 2026-11-24: **Native WebRTC works locally.** `native_environment.mjs` takes a `webrtc` module (node-datachannel 0.29.0 polyfill, installed in `runs/tools/webrtc`), and `native_contributor.mjs` takes `--webrtc-module`. The peer code (`qwen35_peer.mjs`) runs unchanged on it: candidates' `toJSON`, `getStats` with the selected pair, `bufferedAmountLowThreshold`. Test: `tests/webgpu/browser/native_contributor.test.mjs` (4 pass). Local check (`omarchy-20261124-local`, tiny random Inkling GGUF, 3 stages 0,2,4,6, 3 Dawn contributors on the 680M, `tools/local.sh`): relay and webrtc give identical text for chat, code and a 4-message chat (48 tokens each). All 6 directed links opened host-host in 1–3 s; the coordinator saw 150 sends (token reports) against 434 with the relay; requests took 0.40–0.59 s against 0.84–1.04 s.
- 2026-11-24: Gate frozen (above), before renting. It adds a same-machine relay control to the proposed gate, because DO-98's faster decode changed the compute share since report 156.
- 2026-11-24: **Remote run.** User's budget $3 and 150 minutes of machines. Coordinator DigitalOcean `s-8vcpu-32gb-amd` in tor1 (run `do97c`, $0.25/h): GGUF in 3.5 min (sha256 OK), bundle 0,6,18,24 prepared in 2.5 min (sha256 a3ca405b…), coturn 4.x on 3478 with TURN REST credentials. Contributors on Vast.ai in Quebec: RTX A4000 16 GB ×2 (`do97g1`, `do97g2`, $0.131/h each, behind the same public IP) and RTX 4060 Ti 16 GB (`do97g4`, $0.276/h). A first 4060 Ti (`do97g3`) never started ("port is already allocated" on the host) and was deleted by its recorded ID. TCP connect contributor → coordinator 12–17 ms. Code 12f33ea+dirty. Blocks under watch.py (stall 1,500 s, max 4,200 s).
- 2026-11-24: **First attempt stopped after block 2.** In the direct block only A–B opened (srflx-srflx); both links to C were dropped as late, and coturn logged `Cannot find credentials of user <1795520046>`. node-datachannel passes TURN credentials as `turn:USER:PASS@host`, and the TURN REST username `expiry:session:slot` holds colons. libdatachannel URL-decodes both fields, so `native_environment.mjs` now wraps the polyfill's `RTCPeerConnection` and percent-encodes them (`nativePeerConnection`, tested). Checked on the 4060 Ti machine against the real coturn: relay-only gathering gave no candidate raw and a `relay` candidate encoded. The local tiny-model check still gives the relay's tokens (`omarchy-20261124-local/webrtc-encoded`; it replaced the first local webrtc output directory, whose numbers are in the entry above). Block 1-R of this attempt: 255.4 ms per token, tokens as reference.
- 2026-11-24: **Second attempt, all four blocks** (`remote-20261124b`), from the workstation with `tools/drive.sh`:

  | Block | chat 64 tok (s) | code 64 tok (s) | long 2,761 + 64 (s) | first token at 1k (s) | decode at 1k (ms/token) | links |
  |---|---:|---:|---:|---:|---:|---|
  | 1 R | 18.61 | 18.43 | 67.86 | 15.85 | 280.8 | relay |
  | 2 D | 17.02 | 16.08 | 63.12 | 14.95 | 317.5 | 6/6 open |
  | 3 D | 16.80 | 15.77 | 60.87 | 14.78 | 278.8 | 6/6 open |
  | 4 R | 18.37 | 17.45 | 63.95 | 15.61 | 288.2 | relay |

  Links in both D blocks: A–B srflx–srflx (round trip about 2–3 ms), B–C and C–A through TURN on the coordinator (about 23 ms). All opened within 2 s of go on at least one side; B's side of A–B reported 37–40 s (its first attempt was replaced; the pair was open from A's side at 0.4–0.5 s and nothing went to the relay fallback). The polyfill's `getStats` gives `currentRoundTripTime` in milliseconds where browsers give seconds, so the coordinator's route lines show 2000–23000 "ms"; divided by 1,000 above.
- 2026-11-24: **Gate** (`tools/gate.py remote-20261124b`): 1 PASS (all blocks equal report 156's text); 2 FAIL (D 298.1 ms vs R 284.5 ms, ratio 1.048; ≤ 376 ms holds); 3 PASS. Machines deleted by recorded IDs at 10:12 UTC (about 1.1 h; about $0.86 of $3).
- 2026-11-24 (after the report, WS-52 / report 170): the relay arm's contributors reached the coordinator through DO-87's caching `proxy.py`, which delayed every relayed hidden state by about 40 ms (Nagle with the receiver's delayed ACK; fixed since). The relay arm here is therefore about 80 ms per token slower than the relay itself (two hidden-state hops per token); the direct arm was not affected. The gate's verdict stands as measured; the relay-versus-direct comparison should be re-run with the fixed proxy.

## Conclusion

Direct links work for native contributors and keep the tokens, but in one region they do not change decode speed. The relay's three hops per token cost tens of milliseconds at a 13 ms coordinator round trip, while the three consumer GPUs spend about 260–280 ms per token computing: DO-98's decode is 75 ms on an A100, and an RTX A4000 has about 0.29× its memory bandwidth. So the hops are within the 10–40 ms block-to-block noise of a 1k-position decode, and halving time per token was never reachable on these GPUs. The whole-request times show the hop saving is real but small: 64-token chat and code requests were about 10% faster direct, and long prefill about 5%.

What would change the answer: faster GPUs (A100-class stages at 75 ms of compute would make 30–40 ms of hops a third of each token), or contributors far apart (report 156's Delaware–Utah relay cost about 600 ms per token). Neither was tested here.

## Limits

- One region and consumer GPUs; two blocks per arm. The decode figure is a difference of two requests (33 tokens minus 1), and it varied 279–318 ms within an arm.
- Two of three links went through TURN on the coordinator, not peer to peer: the 4060 Ti machine's NAT did not allow a server-reflexive pair. That is still off the coordinator relay (no HTTP polling), but it is one more hop than a true direct link.
- The two A4000 machines shared a public IP, so A–B was probably inside one site.
- The 8k point was not run.

## Reproduce

Workstation: `node --test tests/webgpu/browser/native_contributor.test.mjs`; `runs/tools/webrtc` holds `node-datachannel@0.29.0`; `bash runs/experiments/DO-97/tools/local.sh DIR relay|webrtc PORT` with DIR holding the tiny bundle (`ygaraty model prepare --gguf models/inkling-2-tiny/Inkling-2-tiny.gguf ... --splits 0,2,4,6`), `keys/` and `prompts.json`.

Remote: `tools/make_kit.sh OUT` (src, coordinator and contributor kits). Coordinator (Ubuntu, public IP, UDP open): `coord.sh setup`, `ygaraty init --directory /root/r/keys`, a 32+ character secret in `/root/r/turn.secret`, `coord.sh turn /root/r/turn.secret`. Contributors (Vast.ai ubuntu:22.04, NVIDIA_DRIVER_CAPABILITIES=all): `contrib.sh setup`, `contrib.sh proxy COORD:8000`. From the workstation: `C=... G1=... G2=... G3=... CK=... OK=... bash tools/drive.sh OUT "R D D R"`, then `python3 tools/gate.py OUT`.
