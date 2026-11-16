# Repeated browser precision benchmark (Apple M1)

| | |
|---|---|
| **Date** | not recorded (after the fix above) |
| **Path** | HTTP + browser, 3 modes × 3 runs |
| **Result** | No measurable whole-job slowdown from compensation; plain float32 failed 48 tensors in every run. |
| **Evidence** | `runs/repeated-browser-summary.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

Nine real-browser trials: three runs per mode, two contributors per run. Each
trial resets the same 19,661,824-parameter checkpoint and two-example en→pt-BR
minibatch, tile size 512, fresh Adam step and unchanged PyTorch oracle. Mode order
rotates: float64/plain/compensated; plain/compensated/float64;
compensated/float64/plain. No separate warm-up runs were excluded.

| Mode | Median elapsed (range), seconds | Median worker compute (range), seconds | Correctness |
| --- | ---: | ---: | --- |
| JavaScript float64 | 110.32 (91.87–119.15) | 17.79 (15.35–20.65) | 511/511 strict checks in all 3 runs |
| Original WebGPU float32 | 105.00 (87.63–194.60) | 17.76 (15.38–24.94) | 48 Adam-update tensors fail in every run |
| Compensated WebGPU | 89.94 (82.83–92.99) | 16.88 (16.27–17.09) | 511/511 strict checks in all 3 runs |

The ratio of medians shows compensated WebGPU taking **14.3% less elapsed time**
and **4.9% less worker compute time** than original WebGPU (18.5% and 5.1% less
than float64). These observations do **not** establish that the compensated
shader is faster. The sample is small, the final plain-GPU run was unusually slow,
and the machine is not an isolated benchmark host. In round 2 compensation cost
5.8% more worker compute time. The supported conclusion is **no measurable
whole-job slowdown from the precision fix in this experiment**.

Elapsed timing starts at the first task assignment, after both workers register,
and ends at step completion. It excludes model preparation and adapter setup,
and includes scheduling, HTTP transport and numerical verification. Worker
compute is summed across two workers; it includes all CPU kernels and GPU
conversion, allocation, pipeline creation, dispatch and readback, not isolated
shader time. Fresh SDKs include pipeline creation, although driver caches may
remain warm. Both workers share this machine and GPU.

Every run executed 6,923 browser operations. Both GPU modes executed all 1,344
matmuls on WebGPU, with no CPU fallback. All trials had zero retries/rejections.
HTTP application traffic was approximately 5.318 GB per run in every mode;
tensors remain float64 on the wire. GPU buffers use two float32 components in
compensated mode versus one in plain mode. The maximum compensated Adam delta
error was 9.08e-12 versus 2.26e-5 for original WebGPU, with unchanged tolerances.

Reports: `runs/repeated-browser-benchmark.json` (trials and individual report
paths), `runs/repeated-browser-summary.json` (audited medians, ranges, paired
ratios, fingerprints and methodology). The summary audit verifies matching
reference hashes, optimizer settings, checks, GPU tolerances, two-worker
participation and GPU operation coverage.

To repeat, run `ygaraty experiment benchmark_http --port 8097 --rounds 3`.
Open the printed URL in two real browser tabs; choose Browser 1 and Browser 2,
then click **Start contributor** in each. The harness advances automatically and
releases each completed trial's tensors after both contributors finish. Keep
both tabs open until completion. It uses an isolated in-memory coordinator.
Then run:

```sh
ygaraty experiment summarize_browser_benchmark \
  runs/repeated-browser-benchmark.json \
  --output runs/repeated-browser-summary.json
```
