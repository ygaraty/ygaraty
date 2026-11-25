# Dawn held every uploaded byte until the queue was flushed: settling uploads cuts contributor memory 16.3 → 1.1 GiB at the same load rate

| | |
|---|---|
| **Conclusion** | Failed gate on rate only: memory 16.3 → 1.1 GiB at the same host's rate (30.1 vs 30.5 MB/s), kept (measured) |
| **Status** | executed |
| **ID** | WS-56 (report 182) |
| **Category** | browser |
| **Where** | this machine (memory sampling check); remote (a Vast.ai RTX 3090 24 GB in Utah, `qwen3.8-27b` 1-worker preset served on the host, one native contributor) |
| **Priority** | high |
| **Date** | Added 2026-11-25; finished 2026-11-25 |
| **Path** | `webgpu_kernels.mts` `Gpu.settleUploads` (new): once 256 MiB of `queue.writeBuffer` uploads are pending, submit an empty batch and wait for the previous flush's `onSubmittedWorkDone`; called before each tensor fetch in the contributor's stage load. `?loadTrace=1` now also logs `MEM` lines (resident, heap, array buffers) every 2 s |
| **Result** | Cause (Found): resident memory grew with bytes uploaded (16.2 GiB at 15.3 GB loaded) while the JavaScript heap stayed at 0.02 GiB and array buffers under 0.4 GiB: the rest was Dawn's staging for `writeBuffer`, held because the load submits no work until it finishes. Gate: memory **passes** (peak 1.12 GiB, bound 3); ready (self-test) **passes**; rate **fails** as frozen: 30.1 MB/s against the absolute bound of 52, because this host's own unfixed load ran at 30.5 MB/s (report 180's host did 65). Same host, same model: all pieces at 431 s unfixed, 455 s with the pipelined bound (ready 545 vs 551 s); a first version that waited for each flush before the next download took 550 s (25.9 MB/s) |
| **Evidence** | measured-remote: `runs/experiments/WS-56/vast-ws56h1-20261125/` (`base/`, `fix/`, `fix2/` with `LOAD` and `MEM` lines; `*.log`), `tools/mem.py`; local: `omarchy-local/` |

## Question

Report 180 (DO-105) made a native contributor load the 16.5 GB Qwen3.8-27B in 211 s (65 MB/s) by downloading through core `https`, but its peak resident memory rose from 1.2 to 16.4 GiB: data now arrives faster than it is released. Where does it accumulate (the web stream's queue, completed pieces waiting for upload, uploaded pieces not yet garbage-collected), and can a bound keep memory near the prefetch budget without losing the load rate?

## Prediction

TBD. A stream with a byte-sized high-water mark and the prefetch budget counting bytes in flight should keep the peak to a few GiB at about the same rate.

## Gate (frozen 2026-11-25, before running)

On report 180's kind of host (a rented NVIDIA GPU in the US, the public `qwen3.8-27b` 1-worker preset served on the host, one native contributor, a full cold load): **Works** if the load rate is at least 0.8 × report 180's (≥ 52 MB/s), peak resident memory of the contributor process at most 3 GiB, and the self-test passes. Reported: resident memory over the load split into the JavaScript heap, array buffers and the rest (driver), with and without the change.

## Protocol

1. Locally: a fast source (the coordinator serving local pieces, or a local HTTPS server) to reproduce the memory growth, with resident memory sampled during the load; find where it accumulates; bound it.
2. Remote: one US host with the bound, against report 180's figures; budget from the user first.

Related: [180](180-native-fetch-download-throughput.md), [169](169-spaces-parallel-pieces-crawling-flows.md).

## Log

- 2026-11-25: Added at the user's request, from report 180.
- 2026-11-25: User: "run ws-56". Gate frozen above. Report 180's own numbers suggest memory tracks loaded bytes (1.22 GiB at 0.73 GB loaded before the fix, 16.4 GiB at 16.5 GB after), so the first step is to see whether it is held in JavaScript (array buffers) or by the driver.
- 2026-11-25: Local check (`qwen3.5-0.8b`, AMD 680M, `omarchy-local/`): 0.49 GiB loaded, peak resident 0.25 GiB; no growth on this machine's Vulkan driver, so the remote host was needed.
- 2026-11-25: User's budget $1, 1.5 h. `ws56h1` (Vast.ai 7899206, RTX 3090 24 GB, Utah, $0.215/h, 125 GB RAM). Baseline load T (report 180's code): memory grows with loaded bytes, all outside the JavaScript heap and array buffers (`tools/mem.py`). Fix 1 (wait for each flush before the next tensor): peak 1.2 GiB, 550 s. Fix 2 (wait only for the previous flush, so one flush overlaps the next downloads): peak 1.12 GiB, 455 s. Tests 133 pass. Host deleted by its recorded ID after 0.57 h, about **$0.12**.

## Conclusion

The 16 GiB was not the faster download piling up in JavaScript, as report 180 guessed; it was Dawn keeping a staging copy of every `writeBuffer` until the queue completes work, and the stage load never submitted any. Flushing every 256 MiB and waiting one flush behind keeps the contributor near 1 GiB for a 16.5 GB model, at about the same rate on the same host (−5% to all pieces, −1% to ready). The gate's absolute rate bound failed only because this host downloads at half the speed of report 180's; the change is kept, since it removes a memory need equal to the model size for every native contributor on NVIDIA. The same growth was probably behind report 180's unfixed 1.2 GiB at 0.73 GB loaded.

## Limits

- One NVIDIA host, slower than report 180's; the rate at 65 MB/s with the bound was not measured.
- Browsers (Chrome's Dawn) not checked; the AMD/Vulkan path on this machine showed no growth.
- 256 MiB fixed, not tuned.

## Reproduce

On a US NVIDIA host with this tree: `runs/experiments/DO-62/tools/host.sh setup`, `host.sh launch`, then `loads.py ... T --window 900` (as in the Log); `python3 runs/experiments/WS-56/tools/mem.py OUT/1-T.log`. Unfixed: report 180's `webgpu_kernels.mjs` (without `settleUploads`).
