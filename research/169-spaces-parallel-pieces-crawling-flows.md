# Parallel piece downloads did not speed a cold load from Spaces on a host whose connections to it mostly crawl; loads too slow to finish the gate

| | |
|---|---|
| **Conclusion** | Inconclusive: no load completed; P 1.01 vs S 0.94 MB/s from Spaces, connections mostly crawling (measured, RTX 4090 US) |
| **Status** | executed |
| **ID** | DO-62 (report 169) |
| **Category** | browser |
| **Where** | remote (one Vast.ai RTX 4090 24 GB in California running both the coordinator and a native contributor; pieces from the production Spaces bucket in fra1, read anonymously) |
| **Priority** | high |
| **Date** | Added 2026-11-12; first run 2026-11-14 (one load per arm); rerun 2026-11-24 |
| **Path** | The contributor's Spaces delivery: S `?prefetch=off` (one piece at a time) against P (three speculative pieces plus the demand lane, 30 s no-progress timeout), public `qwen3.8-27b/prepared/v1/1-workers` (16.5 GB, 863 pieces) |
| **Result** | No load completed in either run's window, so by the frozen gate **Inconclusive** (2026-11-14: one full load per arm, S 1,010 s, P 1,050 s, both about 17 MB/s). 2026-11-24: S loaded 1.15 GB in about 1,226 s (0.94 MB/s), P 0.65 GB in about 645 s (1.01 MB/s); a full load would take about 4.5–5 hours. Raw `curl` from that host to the same objects: every response 200, but in a 20 s window one flow moved 50 MB (12.2 MB/s) while three moved 192 B, 1 KB and 138 KB |
| **Evidence** | measured-remote: `runs/experiments/DO-62/vast-do62b-20261124/` (`launch-S1.log`, `launch-P1.log` progress lines, `host.log`); 2026-11-14 run in `runs/experiments/DO-62/a100-20261114-0136/`; local harness check in `omarchy-20261124-local/` |

## Question

Experiment 113's contributor took 45+ minutes to load 44.8 GB from Spaces one piece at a time, and one slow connection stalled the load at 29.6 GB (about 170 KB/s, no timeout). Do concurrent fetches with a stall timeout cut the cold load time and remove stalls, without exceeding the page's memory bounds (each piece is held in JavaScript memory while hashed)?

## Prediction

Load time roughly divided by the concurrency until the host link saturates; no load stuck longer than the timeout.

## Gate (frozen 2026-11-14, before renting)

One rented A100 (Vast.ai, US, outside the bucket's region) joins a production Qwen3-Coder-30B-A3B one-contributor
session (bundle `qwen3-coder-30b/v1/1-workers`, about 18.6 GB in pieces) with a fresh Chrome profile (no cached weights). Arms, by contributor page switch:

- **S**: `--page-param prefetch=off`: one piece at a time (the behavior before 2026-11-13; the 30 s stall timeout stays on);
- **P**: default: three speculative pieces plus the demand lane (256 MiB budget).

Order S P P S S P. Per load: **cold load time** = from the page's first weight request to the `loading` phase
complete (contributor log timestamps), and the renderer's peak memory (`memory_summary.json`).

- **Works** if P's median load time ≤ 0.5 × S's median, no P load has a piece stalled longer than 30 s without
  recovering (no load fails), and P's peak renderer memory ≤ S's peak + 1 GiB.
- **Rejected** if P's median > 0.5 × S's (a load that fails or hangs in P: **Failed gate**).
- **Inconclusive** if fewer than 3 loads per arm complete (e.g. the session does not take a replacement contributor).

## Gate, rerun (frozen 2026-11-24, before renting)

The 2026-11-14 criteria, unchanged: order S P P S S P; **Works** if P's median cold load ≤ 0.5 × S's median, no P load fails or stays stalled past the 30 s timeout, and P's peak memory ≤ S's peak + 1 GiB; **Rejected** if P's median > 0.5 × S's (a P load that fails or hangs: **Failed gate**); **Inconclusive** if fewer than 3 loads per arm complete.

Deviations from 2026-11-14, decided before renting (the production session and its Chrome contributor are not used: the user asked for the production bucket read-only, no deploy and no writes):
- Bundle: the public `qwen3.8-27b/prepared/v1/1-workers` (16.5 GB in 863 pieces) in the production bucket, read anonymously over HTTPS (no credentials; the coordinator gets `url` storage, so contributors download each piece from Spaces themselves).
- Coordinator: `ygaraty launch --model qwen3.8-27b --num-workers 1` on the rented GPU machine itself; the contributor joins it on localhost, so the measured path is Spaces (fra1) to that host.
- Contributor: native (`native_contributor.mjs`, Node + Dawn), which keeps no weight cache, so every load is cold. Memory: the contributor process's peak resident memory (VmHWM) instead of Chrome's renderer.
- Cold load: from the contributor's first weight-file progress line to its all-files line (timestamped by `runs/experiments/DO-62/tools/loads.py`).

## Protocol

This is a historical experiment protocol, not a current command guide. Private
rental and deployment orchestration is archived separately. See the
[quickstart](../docs/quickstart.md) for current launch commands.

Code: bounded concurrency in `weights()`, a stall timeout in `downloadPiece` (abort and re-describe), progress unchanged; unit tests in `tests/` for the timeout and retry. Then a rented GPU in another region than the bucket, a contributor link, Chrome cache cleared between loads.

## Log

- 2026-11-12: Added at user request, from experiment [113](113-kolibri-single-a100-contributor.md)'s findings.

- 2026-11-13: Implemented three speculative downloads (256 MiB assigned-byte budget) plus one demand lane, and 30-second no-progress timeout with one retry. Mocked offline tests cover resource bounds and retry/cancellation. Remote timing and renderer-memory gate remain unrun.
- 2026-11-14: page switch `?prefetch=off|1..8` (fragment or query) on `qwen35_contributor.mjs` (logs `weights_prefetch_<lanes|off>`);
  Cold trials cleared the browser profile before joining; the gate above used
  the same rented A100 as DO-61 part 2 and DO-80's throughput part.
- 2026-11-14: Measurements used a Vast.ai A100 SXM4 80 GB in Texas, a Qwen3-Coder single-worker bundle, relay routing and revision ce3fb50. The ordered trials and timings are recorded below.
- 2026-11-14 01:57: the first fixed-order attempt of load 2 measured nothing: the coordinator still reported the departed contributor as active (90 s heartbeat), so the join was refused (`load2-P-invalid-refused`, discarded). The driver now waits for the session to be vacant (`preparing`, 0 active; 60–90 s) before each join. Load 1 (S) is valid: 16.8 min, about 18 MB/s.
- 2026-11-14 02:13: load 2 (P) ran at about 16.5 MB/s, no faster than S: the host's link from Spaces caps both. Remaining loads would not fit the 2.5 h window together with DO-61; user chose: "Skip loads, run DO-61". DO-62 stops at one load per arm (by the frozen gate **Inconclusive**); DO-80 throughput chats run in load 2's session, then DO-61's six blocks.
- 2026-11-14: results: load 1 (S) 1010 s (16.8 min), load 2 (P) 1050 s (17.5 min); both about 16–18 MB/s, network-bound (rx samples). One load per arm only: by the frozen gate **Inconclusive**, the measured pair shows no gain from parallel pieces on this host. Evidence `runs/experiments/DO-62/a100-20261114-0136/` (gate.json by `runs/experiments/DO-62/gate.py`).
- Shared measured compute cost for DO-61, DO-62 and DO-80: about $1.70 (86 minutes at $1.18/h).
- 2026-11-24: Queued at the user's request ("queue do62, do54, do47, do97 and m8"). Order M-8, DO-97, DO-54, DO-62, DO-47; each remote run up to $3 and 150 minutes of machines, gate frozen before renting. Queued fourth: pieces from the production Spaces bucket, read-only (`~/.config/ygaraty/spaces.env`), no deploy and no writes (user's choice).

## Limits

These results cover the stated hardware, network and run counts only. They do
not establish performance on arbitrary devices or networks. Historical private
harnesses are not distributed as supported public commands.
- 2026-11-24: Rerun harness `runs/experiments/DO-62/tools/loads.py` (native contributor per load, arms by `?prefetch=off`, vacancy wait, VmHWM). Local check (`omarchy-20261124-local`, public `qwen3.5-0.8b` 1-worker preset, 680M): S 123.0 s, P 84.4 s for 0.48 GB in 321 pieces; logs show `weights_prefetch_off` and `weights_prefetch_3`. Gate re-frozen above with its deviations; budget $3 and 150 minutes (queue).
- 2026-11-24: **Rerun.** One Vast.ai RTX 4090 24 GB in California (`do62b`, 61160092, $0.483/h, listed 9.7 Gb/s); `tools/host.sh` setup (Node, Dawn, ygaraty from this tree), then `ygaraty launch --model qwen3.8-27b --num-workers 1` on localhost (public preset: anonymous reads from the production bucket, no credentials, no writes). A single `curl` of one 17.7 MB piece before the run: 7.9 MB/s. Load 1 (S) started 19:22 UTC: after 20.9 minutes 1.15 GB (about 0.94 MB/s), so S was stopped: one S load would take about 5 hours, and three per arm could not fit the 150-minute window, so the gate could only end Inconclusive. One P load then ran for 10.7 minutes: 0.65 GB (about 1.01 MB/s), and was stopped. Raw check right after, four pieces: all HTTP 200; in 20 s, 50,135,040 B (12.2 MB/s), 138,240 B, 1,024 B and 192 B. The host was deleted by its recorded ID after 0.59 h (about **$0.28** of $3).

## Conclusion

The question is still open: neither run finished three loads per arm. What the rerun shows is the failure mode the change was meant for, and that it does not handle it. Connections from this host to the bucket are bimodal: some run at about 12 MB/s, most crawl at bytes to a few hundred KB per second while still returning data. A 30-second no-progress timeout never fires on a crawling flow, and three speculative lanes are mostly crawling too, so P was only about 7% faster than S. A useful next design is a minimum-rate rule (abandon and re-request a piece whose rate stays below a floor, e.g. 1 MB/s over 10 s), measured on the same kind of host.

## Limits

- Partial loads only (21 and 11 minutes); rates from the coordinator's progress lines.
- One host and one provider route to fra1; 2026-11-14's Texas host instead capped every flow together at about 17 MB/s.
- Native contributor, not Chrome; peak memory was not reached in either arm.

## Reproduce

`runs/experiments/DO-62/tools/host.sh setup`, `host.sh launch`, `host.sh loads "S P P S S P"` on a rented GPU with this tree in `/workspace/src`; `tools/loads.py` times each load and its peak memory.
