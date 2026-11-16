# On an A100 host Kolibri-1's resident decode step is GPU-bound (33.1 ms against 32.5 of GPU work): host time is already hidden

| | |
|---|---|
| **Conclusion** | Found: the resident step equals the GPU time (33.1 vs 32.5 ms); host work 11 ms, already hidden (measured, A100) |
| **Status** | executed |
| **ID** | DO-84 (experiment 146) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | One rented A100 80 GB, Kolibri-1 (whole model), production kernel flags; `bench --contexts 64` (non-resident: plan expansion, encoding, GPU time, submit/wait/map per step) and `--resident`, under Chrome (C) and Node + Dawn (N, the same Dawn without Chrome's GPU process, experiment 128); order C1 N1 N2 C2; the host CPU recorded |
| **Result** | Vast.ai A100 SXM4 80 GB (AMD EPYC 7513, driver 595.91.07), Kolibri-1, 1,238 dispatches per token, medians of two runs per runtime. Chrome: plain loop 43.8 ms = GPU 32.5 + expand 2.4 + encode 3.7 + 5.2 unexplained (submission, GPU process, read-back); **resident 33.1 ms**. Node + Dawn: GPU 32.5, expand 4.5, encode 8.5, unexplained 2.8 (about half of Chrome's), plain 48.2, **resident 33.2 ms**. Gate: JavaScript is 54% of Chrome's plain-loop host time (JavaScript-bound by the frozen rule), and Node's unexplained time is 53% of Chrome's (just over half: not attributed to the GPU process). But the question's premise is wrong: the resident step equals the GPU time on this host, so a frozen dispatch list or a native runtime would not shorten Kolibri's decode; its GPU work (the expert kernel, attention) is the lever. Two earlier rentals failed setup on a driver-615.71.09 host (Chrome's GPU process cannot create its context); `pods` gained `YGARATY_VAST_EXCLUDE_DRIVERS`. About $1.07 in all ($0.27 for the two failed hosts). |
| **Evidence** | measured-remote; curated `public/research/evidence/browser/kolibri-host-breakdown-a100-2026-11-13.json`; raw `runs/pods-data/do84/out/`, failed hosts `runs/pods-data/do84/failed-host*.txt`; code d84f0a8 on the machine |

## Question

On A100 hosts the MoE models' resident steps look host-bound (Kolibri: 62.5 ms plain, 35.8 resident, about 28 ms of GPU work), but experiment 145 found the JavaScript side small on the workstation (about 6 µs per dispatch). Is the host time on the rented machines JavaScript on a slower CPU (then a frozen dispatch list helps), or command handling outside JavaScript, such as Chrome's GPU process decoding and validating each dispatch (then fewer dispatches, or a native contributor, help)?

## Gate (frozen 2026-11-13, before renting)

Diagnostic, **Found**, from the medians of each runtime's two runs (base variants at 64 positions):
1. JavaScript host ms = expand + encode; GPU ms; **unexplained ms** = wall − GPU − expand − encode (submission, IPC, the GPU process, read-back).
2. **JavaScript-bound** if expand + encode is at least half of (wall − GPU) in Chrome; **outside-JavaScript-bound** if the unexplained ms is more than half, and also **Chrome's GPU process** if Node + Dawn's unexplained ms is at most half of Chrome's.
3. Reported: resident step per runtime (Node + Dawn as a contributor path for paid machines), dispatches per token, the host CPU.

## Protocol

`runs/pods-data/do84/do84.sh` after `contributor_fleet up do84 --gpu A100_SXM4:79gb` and the Kolibri fetch. User: "approve a100 breakdown" ($2 cap, 90-minute teardown; estimate about $0.60).

## Log

- 2026-11-13: Added and approved at user request, from experiment 145.
- 2026-11-13: local check (680M, Kolibri layers 0–2): Node + Dawn runs the Kolibri bench (wall 11.3 ms, expand 0.4 + encode 0.9, GPU 9.6, 83 dispatches; Chrome 10.8, 0.3 + 0.4, 8.0). Eight layers exceed the 680M's memory under native Dawn (device lost), so the whole-model comparison needs the A100.
- 2026-11-13: first rental (Vast.ai A100 SXM4 80 GB, $1.16/h, driver 615.71.09): setup's WebGPU probe page timed out (`runs/pods-data/do84/failed-host.txt`); machine deleted by recorded ID after about 7 minutes (about $0.14). Retried on a new machine.
- 2026-11-13: second rental failed the same way on a driver-615.71.09 host at the same price (the cheapest A100 80 GB offer); deleted after about 7 minutes. The probe's Chrome log: "GrContext creation failed ... Failed to initialize Skia for SharedContextState". Every successful A100 run today was on drivers 580–610. Added `YGARATY_VAST_EXCLUDE_DRIVERS` (driver prefixes to skip) to the Vast offer filter.
- 2026-11-13 21:40–22:19 UTC: third rental (`do84c`, Czechia, $1.227/h, driver 595.91.07) with 615 excluded: setup passed; Kolibri fetched with a matching SHA-256; C1 N1 N2 C2 all exit 0. Logs copied back; machine deleted by recorded ID at 22:19 (`pods ls`: 0).

## Results

| Per token (median of 2 runs) | Chrome | Node + Dawn |
|---|---|---|
| GPU ms | 32.5 | 32.5 |
| expand + encode (JavaScript) | 2.4 + 3.7 | 4.5 + 8.5 |
| unexplained (wall − GPU − JavaScript) | 5.2 | 2.8 |
| plain loop wall | 43.8 | 48.2 |
| **resident loop wall** | **33.1** | **33.2** |

Gate (by script): JavaScript / (wall − GPU) in Chrome = 0.54 → JavaScript-bound plain loop; Node's unexplained / Chrome's = 0.53 (over the 0.5 bar, so not called Chrome's GPU process).

## What it taught

- The resident loop (experiments 120, 129) already hides all of Kolibri's 11 ms of host work on this host: the step is the GPU time. Experiment 145's frozen-list idea has nothing left to hide here; host time matters only when it exceeds GPU time (slower host CPUs, or much faster GPU work).
- The earlier reading that Kolibri was host-bound on A100s (DO-82's 35.8 ms against "about 28 ms of GPU work") rested on an older GPU profile; today's Kolibri step has 32.5 ms of GPU work. The levers for MoE decode on large GPUs are its GPU kernels: the expert kernel (one workgroup row per slot, 4 rows per workgroup) and attention (the step triples by 1,024 positions, experiment 143).
- Node + Dawn spends twice Chrome's JavaScript time on this host (likely a different V8 tier or the binding's call overhead) but has about half the unexplained time (no GPU-process hop): no net gain for a paid native contributor on this model.
- Rented hosts with NVIDIA driver 615.71.09 cannot run Chrome 157's WebGPU (context creation fails); exclude them until that is resolved.

## Limits

- One host, two runs per runtime, 64 positions only; the GPU profile per kernel family was not taken.

## Reproduce

```bash
YGARATY_VAST_EXCLUDE_DRIVERS=615 python -m ygaraty.experiments.contributor_fleet up do84 --gpus 1 --gpu "A100_SXM4:79gb" --max-hourly 1.6 --minutes 90 --min-ram 64 --disk 200
# fetch the Kolibri GGUF (contributor_fleet fetch), then on the machine:
bash runs/pods-data/do84/do84.sh
```
