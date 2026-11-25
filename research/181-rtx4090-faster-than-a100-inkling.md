# An RTX 4090 decodes Inkling 2 faster than an A100 (22 vs 36 ms per step): the slow 4090 sessions were their hosts, not the GPU

| | |
|---|---|
| **Conclusion** | Found: RTX 4090 21.2–23.8 vs A100 34.7–36.5 ms per step, GPU 15.7 vs 30.4 ms; the 2.7× of reports 171/174 was their hosts (measured) |
| **Status** | executed |
| **ID** | DO-106 (report 181) |
| **Category** | browser |
| **Where** | remote (a Vast.ai RTX 4090 24 GB in Japan with the same CPU model as the A100 of reports 165 and 166, AMD EPYC 7K62) |
| **Priority** | high |
| **Date** | Added 2026-11-25; finished 2026-11-25 |
| **Path** | Inkling 2 Mini whole and resident on one RTX 4090, Node + Dawn, report 166's code; DO-99's steady, call and kernel profiles at about 1k positions |
| **Result** | Steady decode step (run + read, medians of 39 steps): 21.2 and 23.8 ms, against 34.7 (report 166) and 36.4–36.6 ms (report 179) on an A100. Pass profile: GPU 15.7 ms per step against 30.4 on the A100; encode 63.8 ms (6-step average including the first step's compilation, as report 165's 50.2). GPU per kernel (detail, 22.3 ms with its overhead against the A100's 43.8): split attention 3.0 ms, `ink_mix` 2.9, `ink_ctrl_mix` 1.9, the matrix and expert kernels 0.9–1.7 each. Host call costs as on the A100 host (dispatch median 10.5 µs; setPipeline 1.3, setBindGroup 2.2, dispatchWorkgroups 1.9). So the RTX 4090 is about 1.6× faster per step than an A100 here, not 2.7× slower |
| **Evidence** | measured-remote: `runs/experiments/DO-106/vast-do106b-20261125/` (`steady.*.json`, `calls.json`, `pass.json`, `detail.json`, `host.txt`) |

## Question

Report 179 found that splitting Inkling 2 in three costs only 1.14× on one A100, while on rented RTX 4090s stage B ran about 2.05 ms per layer visit against 0.75 on the A100: 2.7×, where memory bandwidth explains about 1.5×. Is the rest GPU time (kernel shapes tuned on the A100, such as `matvecC {16,4}` and `q8FastUpTo`), CPU encode time on the 4090 hosts' processors, or the driver?

## Prediction

TBD.

## Gate (frozen 2026-11-25, before renting; budget $1, teardown 1.5 h)

A diagnostic (`Found`), on one RTX 4090 with the whole model resident at about 1k positions, against the A100 figures of reports 165, 166 and 179 (same code, same tools): steady decode step (40 steps, median, twice), host time per WebGPU call (median per call kind), the pass profile (encode and GPU per step) and the GPU per kernel (detail profile). The result names the largest contributors to the 4090's extra time per step; a fix it points to gets its own gate.

## Protocol

One RTX 4090 (24 GB: the whole 17.6 GB model fits) with DO-101's kit; the same steady-state and per-kernel profiles as reports 165 and 166 at 1k positions; budget from the user first.

Related: [179](179-inkling2-split-cost-one-a100.md), [166](166-inkling2-split-decode-attention.md), [165](165-inkling2-decode-attention-not-host.md).

## Log

- 2026-11-25: Added at the user's request, from report 179.
- 2026-11-25: User: "run 105 and 106 next"; budget $1, teardown 1.5 h. Gate frozen above, before renting.
- 2026-11-25: `do106b` (Vast.ai 60329848, RTX 4090 24 GB, Japan, $0.446/h, EPYC 7K62, driver 595.99.02; a French offer was taken before renting). GGUF in 32 checked parts (13.7 MB/s per connection), then `tools/gpu.sh`: see Result. Host deleted by its recorded ID after 0.68 h, about **$0.30**.

## Conclusion

The premise was wrong: an RTX 4090 runs Inkling 2's decode faster than an A100 (its GPU time per step is about half, plausibly from its higher clocks and larger L2 cache), and with the same CPU its host costs per call are the same. The stage compute of about 155 ms per token in reports 171 and 174 therefore came from those runs' hosts, not from the GPU type: their containers' CPU allowance, or the two stage machines that shared one physical host (80.15.7.37), are the candidates. With report 179's 1.14× for the split, three good 4090 hosts should compute about 25–30 ms per token in all. Choosing hosts by a quick decode benchmark before a session, rather than by GPU name, would avoid the slow ones.

## Limits

- One 4090 host; the slow hosts of reports 171 and 174 were not profiled, so their cause is not identified.
- Whole model only; the split on a 4090 was not run.

## Reproduce

DO-101's kit on a 4090 with 24 GB, then `runs/experiments/DO-106/tools/gpu.sh` (setup and profiles).
