# Handoff reads by region: the extra download is gone; the lost cache hits never cost anything

| | |
|---|---|
| **Date** | 2026-11-01 (follow-up 3 of [experiment 58](058-coordinator-off-the-hop.md)) |
| **Path** | Experiment 59's setup: the tiny Transformer, 8 NumPy workers, non-blocking handoffs, emulated 100/20 Mb/s, 50 ms home links, 12 updates, workstation, no cloud |
| **Result** | Bytes yes, time no. A handoff read now brings only the blocks covering the region each input reads, and the bytes into workers fall from 28.7 to 24.4 MB per update (no handoffs: 23.7). Time is unchanged within the noise: 3.017 → 2.995 s per update (median of 3 runs; no handoffs 3.948). Handing out more unowned tasks per reply restores the tile cache's hits (612 → 10,544) but saves no bytes and is not faster (3.100 s), so the limit stays at 1. Losses are bit-identical in all 12 runs, with 0 retries |
| **Evidence** | `runs/exp60/r{1,2,3}-home-{h0,f,r,r4}/profile.json`, captures `runs/exp60/cap-{h0,f,r,r4}-cap/` (local, not in git); `relay/relay_test.go: TestHandoffSendsOnlyTheBlocksTheRegionsCover`; `tests/test_handoff.py: test_a_read_needs_only_the_blocks_that_cover_it` |

## Question

Experiment 58 left two costs of handoffs unmeasured.
- **Lost cache hits:** handed tasks hit the worker's tile cache far less, 132 hits against 7,500–12,400.
- **Unknown bytes:** the bytes workers read from the relay were not counted.

Do handoffs cost bytes or time, and where?

## Diagnosis

Temporary counters on the coordinator and a capture event on the worker (`worker.handoff_bytes`: bytes fetched against bytes covering what was read) gave two answers.

- **The cache hits were lost to one rule, not to the cache.**
  - Both arms key the same 18,003 `embed_rows` inputs over 6 updates. With handoffs, only 533 of them hit the cache instead of 5,268.
  - Without handoffs, a worker takes several `embed_rows` tasks in one reply, and the later ones read tiles that the earlier ones just cached.
  - Experiment 58 hands out only one unowned task per reply, so that sharing is lost. The rule was there because experiment 58's worker blocked on handed tasks; since experiment 59 it no longer does.
- **The extra bytes were whole results.**
  - A handoff read brought every block of the producer's result, even for an input that reads one row of it. A `tf_owned_step` reads a 1-row region of a 376-row gradient, for instance.
  - Of 19.4 MB of handoff reads per update, 15.0 MB covered what was read.
  - A committed input read the usual way gets only its covering blocks (`store.locate`).

## What changed

- **Region-limited reads.** The worker works out, from each input's recipe and region, which part of each named result it reads (`storage.relay.recipe_needs`, which follows `assemble` and `transpose`), and which committed blocks it reads.
  - `POST /handoff/get` takes `regions` per name and sends only the blocks that overlap them. It still waits for every name, since the coordinator checks every attempt.
  - The relay reads block regions from the manifest the producer published.
  - `YGARATY_HANDOFF_FULL=1` keeps experiment 59's whole-result reads, as the control here.
- **The unowned limit is a setting.** `Handoff.HANDOFF_UNOWNED` is 1 by default; `YGARATY_HANDOFF_UNOWNED` overrides it. It now counts only the unowned handed tasks in a reply; experiment 58's rule allowed no unowned task once any task had been handed in that reply. All handoff arms here, including f, use the new count.

## Design

Four arms, 3 runs each on home links. The order rotated across repeats (h0 f r r4, then r4 r f h0, then f r4 h0 r), and each run started on an idle machine:
- **h0:** no handoffs;
- **f:** experiment 59, whole-result reads;
- **r:** region-limited reads;
- **r4:** r with up to 4 unowned tasks per reply.

Then one captured run per arm (8 updates), for the bytes.

The bytes into workers are counted on the workers' side, from the captures:
- the claim replies' bodies (including the inputs pushed with them);
- the handoff reads.

The workers made no relay reads of their own; their other inputs arrive pushed. The relay's own served-bytes counter, tried first, also counts the coordinator's reads and varied by 9 MB between identical runs, so it is not used.

## Results

Seconds per update (median of updates 2 to 10):

| Arm | Run 1 | Run 2 | Run 3 | **Median** | Cache hits per run |
|---|---|---|---|---|---|
| h0 | 3.901 | 3.948 | 4.021 | **3.948** | 10,636–15,266 |
| f (experiment 59) | 3.001 | 3.121 | 3.017 | **3.017** | 132–1,035 |
| r | 3.046 | 2.995 | 2.942 | **2.995** | 612–644 |
| r4 | 3.162 | 3.100 | 2.988 | **3.100** | 7,497–12,385 |

Bytes into workers per update (captured runs):

| Arm | Claim replies | Handoff reads | Of which covering what was read | **Total** |
|---|---|---|---|---|
| h0 | 23.7 MB | — | — | **23.7 MB** |
| f | 9.3 MB | 19.4 MB | 15.0 MB | **28.7 MB** (+21%) |
| r | 9.6 MB | 14.7 MB | 14.7 MB | **24.4 MB** (+3%) |
| r4 | 9.7 MB | 14.4 MB | 14.4 MB | **24.1 MB** (+2%) |

All twelve timed runs give the same twelve losses, with 0 retries and 0 rejections.

## What it means

- **Handoffs no longer cost bytes.** With region reads, workers download about what they do without handoffs (+3%, against +21% before). The bytes move from the coordinator's claim replies to the relay.
- **They never cost time.** The 4.3 MB per update removed is about 0.54 MB per worker, roughly 45 ms of a 100 Mb/s downlink spread over a 3 s update. It mostly travelled off the chain. f and r differ by 0.7%, inside the spread of either arm.
- **The lost cache hits did not matter.** r4 brings the hits back, but each hit saved a one-row tile whose block came in the same reply anyway. So bytes do not move (24.1 against 24.4 MB). r4 also has the highest median of the handoff arms (3.100 s), though its runs (2.99–3.16 s) overlap the others.
- **Defaults:** region reads stay on, and the unowned limit stays at 1.
- **Overall:** handoffs now take 23–24% off the tiny model's update on home links (experiments 59 and 60), with the same bytes into workers and bit-identical losses.

## Limits

- Experiment 58's limits apply: emulation, the tiny model, one machine.
- Bytes come from one captured run per arm.
- On a downlink-bound pool the 21% would have shown in time. Here the 100 Mb/s downlinks have room, so these runs cannot show what region reads are worth on a slower downlink.

## Reproduce

As in [experiment 58](058-coordinator-off-the-hop.md), with `--handoff`. Add `YGARATY_HANDOFF_FULL=1` for arm f and `YGARATY_HANDOFF_UNOWNED=4` for r4. The local drivers are `runs/exp60/campaign.sh`, `precise.py` (time) and `bytes.py` (bytes from the captures); the campaign takes about 10 minutes.
