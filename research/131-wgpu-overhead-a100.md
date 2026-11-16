# Deno with wgpu loses its 37 ms per step waiting for completions: about 11 ms per mapAsync and per wait, a polling delay

| | |
|---|---|
| **Conclusion** | Found: 58% of Deno's extra time is mapAsync (11 ms each), 40% onSubmittedWorkDone; polling, not GPU (measured, A100) |
| **Status** | executed |
| **ID** | DO-73 (experiment 131) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | med |
| **Date** | Added 2026-11-13 |
| **Path** | Qwen3.5-9B on a rented GPU with more VRAM than the model (wgpu allocates storage from VRAM only; the 680M's 3 GiB cannot hold it, experiment 128): Deno + wgpu with host-side timing of submit, `mapAsync` and `onSubmittedWorkDone`, against Node + Dawn; then the fix (e.g. a read-back pattern wgpu prefers) and a rerun |
| **Result** | Qwen3.5-9B on an A100 SXM4 40 GB, one decode step at 64 positions (medians of two runs): Deno + wgpu 72.0 ms against Node + Dawn 34.4 (Chrome 36.4). Of the 37.6 ms difference, `mapAsync` holds 21.9 (58%: 22.3 ms for 2 maps, about 11 ms each, against 0.45 under Dawn), `onSubmittedWorkDone` 14.8 (40%; GPU time is only 4.1 ms longer, 26.0 against 21.9), submit 1.7. So about 30 ms per step is completion polling in Deno's binding (a ~10 ms tick), not GPU work. Tokens identical across all three runtimes at 64 and 1,024. Same rental: experiment 129's resident loop on the A100 (see its addendum). About $0.31. |
| **Evidence** | measured-remote; curated `public/research/evidence/browser/wgpu-overhead-a100-2026-11-13.json`; raw `runs/pods-data/do73/out/`; code a9e7ecd |

## Question

Experiment 128 on an A100: under Deno + wgpu a step takes 71.5 ms with 26.0 ms of GPU work, against 34.5 and 21.9 under Node + Dawn, while wgpu's copy probe reaches 1,336 GB/s (Dawn 745). Where do the 37 extra ms go (read-back polling, submission, validation), and how fast is our engine under wgpu once they are removed?

## Prediction

Most of the gap is `mapAsync` latency (polling in Deno's binding); removed, wgpu's step is at least as short as Dawn's, and its faster memory path shortens the GPU time once the kernels are tuned for it.

## Gate (frozen 2026-11-13, before renting)

Diagnostic, **Found**. One Vast.ai A100 (user: "approved", $2 cap, 90 min, shared with experiment 129's A100 check). Qwen3.5-9B, `bench --contexts 64,1024 --matvec 2,4,8 --matvec-c 16,4` under Chrome (C), Node + Dawn (N) and Deno + wgpu (D), order C N D D N C; every step records host ms in `queue.submit`, `onSubmittedWorkDone` and `mapAsync` (with their counts) beside wall, encode, expand and GPU ms.
Reported per runtime: the medians of each phase at 64 and 1,024 positions, and which phase holds Deno's extra time over Dawn (the phase with the largest difference, and its share of the wall-time gap). A fix is not part of this run; one found here becomes its own experiment.
Experiment 129's A100 check on the same machine: `bench --contexts 64,1024 --resident --matvec 2,4,8 --matvec-c 16,4` under Chrome and under Node + Dawn (B R R B arms), tokens identical and ms per token per arm reported in 129's file as an addendum.

## Protocol

A host-timing variant in the bench (timestamps around submit, `onSubmittedWorkDone`, `mapAsync`). A small model on a discrete GPU first if one is available, then the A100 with DO-70 and WS-43's measurements; budget asked first.

## Log

- 2026-11-13: Added at user request ("Add: profile wgpu overhead"), from experiment [128](128-native-webgpu-a100.md).

- 2026-11-13: budget approved ("approved": $2 cap, 90-min watchdog, one A100 shared with experiment 129's A100 check). Host timing added to `Gpu` (`gpu.host`: ms and counts in submit, wait, map) and printed by `bench`.

- 2026-11-13 01:0x UTC: rented `do73` (A100 SXM4 40 GB, $0.589/h, 90 min). First pass failed in set-up: the model fetch timed out at 3.8 of 5.6 GB, and Deno's `apt-get` collided with Node's (dpkg lock); every arm then failed at once. Rerun on the same machine (`do73b.sh`: fetch with 3 tries, Deno's install after Node's): all steps exit 0. Results copied to `runs/pods-data/do73/out/`; `pods down do73`; 0.53 h, **$0.31**.

## Results

ms per step at 64 positions (1,024 in brackets), medians of two runs:

| | Chrome | Node + Dawn | Deno + wgpu | Deno − Dawn |
|---|---|---|---|---|
| wall | 36.4 (39.0) | 34.4 (34.9) | 72.0 (74.0) | +37.6 |
| GPU | 21.9 | 21.9 | 26.0 | +4.1 |
| `onSubmittedWorkDone` wait | 27.3 | 22.4 | 37.3 | +14.8 (40%) |
| `mapAsync` (2 per step) | 1.15 | 0.45 | 22.3 | +21.9 (58%) |
| `queue.submit` (2 per step) | 0.10 | 1.06 | 2.80 | +1.7 (5%) |
| encode | 5.7 | 6.7 | 5.9 | −0.9 |

Tokens identical in all six runs at both contexts. **Found.**

## What it taught

- Deno's WebGPU binding resolves completions late: each `mapAsync` takes about 11 ms even for a 4-byte read that is already done, and `onSubmittedWorkDone` returns about 11 ms after the GPU work ends. That is the signature of a fixed polling tick, not of wgpu's GPU path.
- Under Dawn, Chrome's GPU-process hop costs about 5 ms of waiting per step against Node (27.3 against 22.4 ms).
- A wgpu runner without that tick (Rust, or a Deno build or flag that polls the device eagerly) should approach Dawn's step, with wgpu's faster copies (1,336 GB/s, experiment 128) to gain once our kernels use them; wgpu's GPU time here is still 19% longer than Dawn's for our kernels.

## Limits

- Deno 2.9.8 only; whether a setting removes the tick, or a Rust wgpu runner avoids it, was not tested.
- One host; the first pass lost to set-up failures.

## Reproduce

```bash
# A100 after contributor_fleet setup: runs/pods-data/do73/do73.sh (do73b.sh: the same with the fetch retried and the installs in sequence)
```
