# With fast stages, Inkling 2's three-contributor tokens cost about 60 ms per hop over the relay, and TURN-relayed data channels were no faster

| | |
|---|---|
| **Conclusion** | Failed gate: direct links 313.9 vs 216.1 ms per token at 1k over the relay; all links through TURN (measured, 3 fast GPUs) |
| **Status** | executed |
| **ID** | DO-100 (report 167) |
| **Category** | browser |
| **Where** | remote (DigitalOcean coordinator + coturn in sfo3; two RTX 4090 24 GB and one A100 SXM4 40 GB on Vast.ai in California) |
| **Priority** | med |
| **Date** | Added 2026-11-24; started 2026-11-24; finished 2026-11-24 |
| **Path** | Report 164's setup with report 166's split attention and report 165's host path in both arms: layers 0–5 \| 6–17 \| 18–23, `f32` wire, native contributors with `--webrtc-module`, blocks R D D R on the same machines |
| **Result** | Item 1 passes: chat, code and long give report 156's 64 greedy tokens in all 4 blocks. Item 2 **fails**: decode at 1k, R 211.0 and 221.1 ms per token; D 313.9 in block 3, and block 2's measurement is invalid (its 1-token request took 29.2 s, longer than the 33-token one, while B re-made its link to A). Item 3 **fails**: block 2 sent hidden states through the relay for a while (298 sends and 112 receive polls against 235 and 49 in the clean block 3); in both D blocks B's side of the A–B link opened only after 68–70 s. Every link used a TURN relay candidate on at least one side (round trips 18–25 ms). Whole 64-token requests: D 16.1–18.7 s, R 16.1–19.8 s |
| **Evidence** | measured-remote: `runs/experiments/DO-100/remote-20261124/` (per block `N-X.json`, `N-X.serve.log`, contributor logs; `setup.txt`); the stalled first attempt in `remote-20261124-g3stall/` |

## Question

Report 164 (DO-97) found direct links keep the tokens but do not change decode speed in one region on consumer GPUs: about 270 ms of compute per token against 30–40 ms of hops, so the hop saving sat inside the noise (298 vs 285 ms per token; 64-token requests about 10% faster). Where the hops are a large share of each token, do direct links pay off?

- **Fast stages:** A100-class GPUs, where report 162's decode is 75 ms per token on one GPU, so three relayed hops at a 13 ms round trip could be a third of each token.
- **Far apart:** contributors in two regions (report 156's Delaware–Utah relay cost about 600 ms per token).

## Prediction

TBD. With fast stages in one region, direct about 0.7× the relay. Across regions, the relay pays two coordinator legs per hop against one path direct, so direct about 0.5–0.7× the relay.

## Gate (frozen 2026-11-24, before renting)

Arm: fast stages in one region (three RTX 4090 24 GB in California; coordinator with coturn on DigitalOcean in San Francisco). Code: report 166's split attention and DO-99's host path in both arms. Blocks R D D R on the same machines, as report 164.

1. **Same tokens:** in every block, chat, code and long give report 156's 64 greedy tokens (decoded text).
2. **Faster:** D's decode at 1k positions (33-token minus 1-token request, as report 164) at most 0.75× R's (median of each arm's two blocks).
3. **Links:** all three pairs open in every D block, none dropped, no hidden states through the relay (receive polls per block, as report 164). Reported: candidate types and round trips (now in seconds), whole-request times.

## Protocol

As report 164 (`runs/experiments/DO-97/tools`). Pick one arm first (fast stages in one region is cheaper to interpret); budget and teardown deadline from the user first. Also make the native contributor report round-trip times in seconds (the node-datachannel polyfill gives milliseconds), so the route lines read correctly.

Related: [164 (DO-97)](164-inkling2-direct-links-three-contributors.md), [156 (DO-95)](156-inkling2-mini-one-gpu-three-contributors.md), [162 (DO-98)](162-inkling2-fewer-dispatches-a100.md), DO-99.

## Log

- 2026-11-24: Added at the user's request, from report 164's conclusion.
- 2026-11-24: Prioritised after DO-99 (user: "prioritise 99 now then 100"); DO-101 ran first at the user's choice. Budget $3, machine deadline 2.5 h (watchdogs set to 100 minutes so three 4090s and the coordinator stay under $3). The native contributor now reports round trips in seconds (`nativePeerConnection` divides libdatachannel's milliseconds; tested). Gate frozen above, before renting.
- 2026-11-24: **Remote run.** Coordinator DigitalOcean `s-8vcpu-32gb-amd` in sfo3 (`do100c`, $0.25/h): GGUF, bundle (sha256 a3ca405b…, as report 164) and coturn in 7 minutes. Contributors on Vast.ai, California: RTX 4090 24 GB ×2 (`do100g1`, `do100g2`, $0.483/h each, one public IP, about 10 ms from the coordinator). The third, an RTX 4090 48 GB (`do100g3`, $0.511/h), stalled at 43 of 139 weight files: its provider answers TCP connects in under 1 ms and HTTP in 0.4–0.7 s, a proxy in front of the host. A replacement RTX 3090 (`do100g4`) showed the same signature with the same provider and was deleted unused. The third contributor became an A100 SXM4 40 GB (`do100g5`, $0.544/h, 15 ms from the coordinator, no proxy). All machines deleted by their recorded IDs; about **$2.27** of $3.
- 2026-11-24: **Blocks** (`runs/experiments/DO-97/tools/drive.sh`, R D D R):

  | Block | chat 64 tok (s) | code 64 tok (s) | long 2,761 + 64 (s) | 1-token at 1k (s) | 33-token at 1k (s) | decode at 1k (ms/token) | relay sends / receive polls |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 1 R | 19.82 | 19.68 | 33.75 | 5.17 | 11.92 | 211.0 | 824 / 824 |
  | 2 D | 16.52 | 16.08 | 37.42 | 29.19 | 14.21 | invalid | 298 / 112 |
  | 3 D | 18.72 | 18.38 | 32.70 | 6.47 | 16.51 | 313.9 | 235 / 49 |
  | 4 R | 16.54 | 16.10 | 27.20 | 4.86 | 11.94 | 221.1 | 824 / 815 |

  Links in both D blocks (round trips now in seconds, shown in ms): A–B prflx–relay (18–21 ms), B–C and C–A srflx/prflx–relay (24–25 ms). No pair found a peer-to-peer path.
- 2026-11-24: **Gate** (frozen above): 1 PASS. 2 FAIL (only valid D value 313.9 against R's median 216.1; ratio 1.45). 3 FAIL (block 2's relay fallback; late opening of B's side of A–B in both D blocks). DO-97's `gate.py` would print PASS here, because it takes the median of block 2's negative value; it needs an invalid-measurement check before reuse.
- 2026-11-24 (after the report, WS-52 / report 170): the relay arm's contributors reached the coordinator through DO-87's caching `proxy.py`, which delayed every relayed hidden state by about 40 ms (Nagle with the receiver's delayed ACK; fixed since). The relay arm here is therefore about 80 ms per token slower than the relay itself (two hidden-state hops per token); the direct arm was not affected. The gate's verdict stands as measured; the relay-versus-direct comparison should be re-run with the fixed proxy.

## Conclusion

With fast stages, the three-contributor session spends far more per token than its compute. One A100 decodes in about 35 ms (report 166); here the relay arm took 211–221 ms per token at 1k, about 60 ms per hop at round trips of 10–15 ms. So most of each hop is not network time: it is the contributor's own work around a hop (reading the hand-off back from the GPU, sending it, waiting for the next stage) or how the hand-off is scheduled. Direct links did not help, because on these providers every link went through TURN on the coordinator, and the data channels over TURN were slower than the relay in the one clean block. A larger saving needs the per-hop cost understood first, measured on one machine where the network is out of the picture.

## Limits

- One clean D block; block 2's timing is invalid and its links fell back for a while.
- Mixed GPUs (two RTX 4090s, one A100), and the two 4090s shared a public IP; R's two blocks differ by 16% on whole requests (19.8 vs 16.5 s for chat), so whole-request differences between arms are within noise.
- No peer-to-peer candidate pair formed; this does not measure direct links on hosts with open UDP.

## Reproduce

As report 164 (`runs/experiments/DO-97/tools`: `make_kit.sh`, `coord.sh`, `contrib.sh`, `drive.sh`), with this tree's code. Read block timings by hand or guard `gate.py` against negative decode values.
