# Experiment 45: ConvNet remote scaling — WebGPU versus native PyTorch

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Name** | `convnet_remote_scaling` |
| **Path** | HTTP through SSH tunnels or a self-signed HTTPS proxy (`scripts/remote.sh https-proxy`), one full DiLoCo replica per physical machine; separate WebGPU and PyTorch cases. Measured so far: coordinator on a DigitalOcean g-8vcpu-32gb (sfo2), one Apple M5 in Chrome 154 as the WebGPU worker; code at 3af9c87 for the gate and first case, 64b7f48 and 96da594 for the wire and pixel cases |
| **Result** | One machine (an Apple M5), one run per case. The full-model gate passed across machines: the M5's WebGPU checkpoint after 3 rounds matches native PyTorch CPU float64 to 1.5e-6 (tolerance rtol .003, atol 2e-5). On one coordinator, at batch 8: H=5 ran at 5.99 images/s with the float64 wire, 7.69 with `--wire-dtype float32`, 8.13 adding `--pixels uint8`; H=20 at 20.91 (float32 wire) and **25.96** (float32 + uint8 pixels, with the M5's compute rate unchanged, so the gain is transport); H=50 dropped to 17.37 as the M5's compute slowed threefold. The 2/4/8-machine matrix, repeats and the native MPS baseline are not run |
| **Evidence** | Remote reports: [`experiments/evidence/diloco/convnet-remote-scaling-2026-10-28/`](evidence/diloco/convnet-remote-scaling-2026-10-28) (gate-cpu, gate-webgpu-b, webgpu-m1-h5-r1, and the six `dw45-*` wire/pixel cases); checkpoints in `runs/pods-data/{web,dw45}/server/runs/experiments/045/` (not synced). Local smoke reports/checkpoints: `runs/convnet-remote-scaling-smoke/{native-2,webgpu}/`; checkpoint comparison: `parity.json`. These are generated, ignored files. |

## Question and fixed protocol

Can adding independent GPUs make portable browser training competitive with native PyTorch, measured in **aggregate images/second**? Run each backend separately at **1, 2, 4, and 8 physical machines**. Keep the same machines available in the same order across backends. One worker per host; use each host's own GPU. A second tab on the same GPU is not another physical worker.

The full model has **94,538 parameters**: three same-padded 3×3 convolutions with channels 32/64/128, ReLU, max pooling after the first two layers, global average pooling, and a ten-class head. Inputs are 3×32×32, normalized to [-1,1]. It is the canonical SmallConvNet architecture, with the portable recipe's parameter names/layout. Native workers use PyTorch convolution, autograd and Adam, independently of the portable kernels. No `torch.compile` or mixed-precision autocast.

Defaults: seed 2026; batch **8 per replica** (portable limit: 16); inner Adam learning rate .001; outer Nesterov learning rate .7, momentum .9; **5 local steps**, **2 warmup rounds**, **10 measured rounds**. Repeat the matrix at H=20 and H=50 only after H=5 completes. Each H is a separate comparison. This is weak scaling: adding replicas also adds examples processed per round. Repeat each case three times in a fresh directory, preserving failures, and report the median across completed runs as well as individual results.

Use PyTorch **CPU float64** for the repository's canonical precision baseline, **CPU float32** as the native CPU alternative, and **CUDA/MPS float32** as the native GPU baseline. MPS cannot use float64. WebGPU uses float32 convolution/pooling/matrix kernels and JavaScript float64 Adam/other tasks; the coordinator is float64. Record precision, host model, GPU, OS, browser/PyTorch versions, thread count and network path. Use the same thread count in native cases; CPU threads default to 1 per worker.

Primary metric = measured examples / elapsed coordinator time from preparation of the first measured round through the last measured outer update. This includes graph compilation, JSON encoding/decoding, weight/input transfers, synchronization, stragglers and inter-round polling. Initial membership and warmup are excluded. Native workers receive compact input batches; browser workers receive compiled plans. This measures the current end-to-end implementations, including their different transport overhead, rather than isolated GPU kernel performance.

Reports additionally preserve every round's time, outer update time, each worker's local compute duration, GPU dispatch counts, membership metadata, traffic bytes, data/source hashes and final weights. Timers await downloads; they are wall timers, not GPU timestamp queries. `active_round_seconds` excludes gaps between rounds; primary `measured_seconds` includes them. The difference is synchronization/polling overhead, not a pure network-time measurement. A worker's compute time includes weight rebasing, execution and device readback, but excludes JSON parsing and transfers over HTTP.

## Results so far (2026-10-28)

The coordinator ran on a DigitalOcean g-8vcpu-32gb Droplet in sfo2 at commit 3af9c87 (`source_dirty: false`), with PyTorch 2.15.0 CPU for the preflight fixture. The worker was one Apple M5 in Chrome 154 (WebGPU adapter `apple`/`metal-3`, not a fallback adapter), reaching the coordinator through `scripts/remote.sh https-proxy` (self-signed TLS on port 8443 to 127.0.0.1:8000) over the public internet, not an SSH tunnel. The M5 passed all 42 preflight checks before each case.

**Full-model gate** (1 replica, batch 2, H=1, no warmup, 3 rounds, synthetic). The native side was a PyTorch CPU float64 worker on the coordinator's Droplet (`--threads 1`); both cases used seed 2026 and the same synthetic batches. Every array of the final `weights.npz` passed `assert_allclose(rtol=.003, atol=2e-5)`; the largest absolute difference is 1.5e-6, in `conv2.weight`.

| Round | Native CPU float64 loss | M5 WebGPU loss |
|---:|---|---|
| 1 | 2.3197080689 | 2.3197080524 |
| 2 | 2.2452896753 | 2.2452896670 |
| 3 | 2.4276214287 | 2.4276213885 |

**First measured case, `webgpu-m1-h5-r1`** (1 replica, batch 8, H=5, 2 warmup + 10 measured rounds, synthetic): **4.87 images/s** (400 images in 82.1 s measured; 71.3 s inside rounds). Per measured round, medians:

| | Seconds |
|---|---:|
| Round at the coordinator (40 images) | 7.1 |
| M5 compute: rebase, 5 local steps, readback | 2.86 |
| Active-round time outside the worker compute timer: compilation, JSON handling, HTTP/TLS transfers and coordination, not separately profiled | about 4.2 |
| Gap between rounds: result acknowledgments, report writing, HTTP/scheduling and possible polling, not in active round time | about 1.1 |

Each round ran 300 CPU tasks (JavaScript float64 Adam and the other non-convolution operations), 100 GPU tasks and 15 matrix products: per local step, 60 CPU and 20 GPU tasks, with an upload and a readback per GPU operation. Even the compute alone is about 14 images/s. On this setup most end-to-end time is outside the worker compute timer. The current path includes plan encoding, JSON transport, CPU tasks and per-operation device transfers, but these measurements do not isolate their individual costs or establish the GPU's arithmetic utilization. The training loss stays near ln 10 because the labels are synthetic and random: this case measures throughput, not learning.

### Client perspective and reconciliation with the server

The Chrome worker `mac-m5-01` joined the public HTTPS endpoint on port 8443, passed the hardware preflight, executed all 12 displayed versions without a client error and received the final completion acknowledgment. The browser reported 100 GPU tasks per version, 1,200 including warmup. Version numbers are zero-based: versions 0 and 1 are the two warmups; versions 2–11 are the ten measured rounds. A screenshot showing 12 completed versions therefore does not imply 12 measured rounds.

The displayed **13.39–16.65 local images/s** and the server's **4.87 end-to-end images/s** refer to different intervals. Cross-checking all 12 rows against the preserved coordinator report reproduces every displayed local rate to its two-decimal rounding, every displayed final loss to six decimals, and every GPU count. The coordinator stores the compute duration supplied by the browser; this is a reporting/aggregation cross-check, not an independent measurement of GPU compute.

For the **same 400 measured images**, excluding both warmups:

| Timing boundary | Elapsed seconds | Images/s |
|---|---:|---:|
| Sum of worker `trainer.runRound()` timers | 27.9144 | **14.3295** |
| Sum of coordinator active-round timers | 71.3391 | **5.6070** |
| First measured round preparation through final measured commit, including gaps | 82.0586 | **4.8746** |

Thus there is no contradictory throughput measurement: 43.4247 s falls inside coordinator rounds but outside the worker compute timers, and another 10.7195 s falls between active rounds. Worker compute is 34.0% of the complete measured wall interval. End-to-end throughput is about one third of the compute-only rate. Rates are calculated as total images divided by total elapsed seconds, not as the arithmetic mean of per-round images/s.

The worker timer starts after the `/round` response has been downloaded and parsed and stops before `/result` JSON serialization/upload. It includes weight rebasing, CPU tasks, GPU dispatches and device readbacks. The active coordinator timer starts before batch generation and plan construction and ends after the outer update commits. Gaps can include result acknowledgment construction, report I/O, HTTP round trips, JSON parsing, scheduling and polling. The three clocks therefore form nested intervals in this one-worker case, but the residual is **not a measured pure-network duration**. No cross-machine timestamp subtraction is used: each side measures elapsed durations with its own monotonic clock.

Traffic counters report **54.02 MB sent and 24.76 MB received for the entire case**, including warmups and protocol messages. They are application payload counters, not on-wire TLS byte counts or measured-only traffic. Dividing these totals by ten measured rounds would incorrectly assign warmup traffic to the measured rounds; the earlier 5.4 MB down / 2.5 MB up per-round interpretation has been removed.

The earlier client-only summary deliberately reported all observed rounds and its local throughput range. The coordinator report is the authority for warmup selection and end-to-end throughput. The primary performance result remains **4.87 images/s**, not 14.33 images/s. The local number is useful for locating overhead and sizing future work; it is not a replacement for the experiment's end-to-end metric.

Reconciled per-version observations and calculations are preserved in [`client-observations.json`](evidence/diloco/convnet-remote-scaling-2026-10-28/client-observations.json). The original client log and start/progress/completion screenshots remain locally under `runs/exp45-remote-run2/` (ignored generated artifacts). Neither the evidence JSON nor this write-up includes coordinator credentials.

For scale only, not as a comparison: in the gate the native CPU float64 worker on the Droplet ran 160–225 local images/s at batch 2. The fixed protocol's native baseline for this worker is PyTorch MPS float32 on the same M5, which has not been run.

A first gate coordinator (`gate-webgpu`) was stopped before any worker joined, to raise `--timeout` above the default 900 s; it has no report.

### Wire dtype, pixel encoding and H (2026-10-28, second coordinator)

A fresh coordinator (DigitalOcean g-8vcpu-32gb, sfo2) served six one-worker cases to the same M5 and Chrome, one case at a time through `scripts/remote.sh https-proxy`; all use batch 8, 2 warmup and 10 measured rounds, synthetic data, seed 2026. `--wire-dtype float32` (64b7f48) sends weights and pseudo-gradients as base64 float32; `--pixels uint8` (96da594) sends each step's images as uint8 pixels that the browser normalizes bitwise identically. The uint8 cases ran from a second checkout at 96da594 so that cases already waiting kept their code.

| H | Wire | Pixels | End-to-end images/s | M5 compute images/s | Compute share of measured time | Coordinator sent / received, whole case |
|---:|---|---|---:|---:|---:|---:|
| 5 | float64 | float64 | 5.99 | 31.2 | 19% | 54.0 / 24.8 MB |
| 5 | float32 | float64 | 7.69 | 48.6 | 16% | 36.0 / 6.1 MB |
| 5 | float32 | uint8 | 8.13 | 41.2 | 20% | 9.1 / 6.1 MB |
| 20 | float32 | float64 | 20.91 | 79.0 | 27% | 125.8 / 6.1 MB |
| 20 | float32 | uint8 | **25.96** | 77.8 | 33% | 18.1 / 6.1 MB |
| 50 | float32 | float64 | 17.37 | 25.4 | 68% | 305.5 / 6.1 MB |

- **Local steps per round matter most:** H=5 to H=20 is 2.7x at the same encoding. The fixed per-round cost (round trip, plan compilation on the coordinator, rebasing) is spread over four times the work.
- **uint8 pixels: +24% at H=20,** where the M5's compute rate was unchanged (79.0 against 77.8 images/s), so the gain is transport: 126 MB down became 18 MB. At H=5 it is +6%, within the run-to-run variation below.
- **float32 wire: +28% at H=5** against the float64 control on the same coordinator. The upload falls fourfold.
- **H=50 is slower than H=20** because the M5's own compute rate fell to 25 images/s. That case still used float64 pixels, about 25 MB of plans per round held by the tab. Memory pressure, garbage collection or sustained-load throttling are possible causes; none was tested, and H=50 was not rerun with uint8 pixels.
- **Run-to-run variation is large:** the M5's compute rate at H=5 was 14.3 (first coordinator), 31.2, 48.6 and 41.2 images/s in four runs. Its timer excludes transport and JSON parsing, so this is the M5 and browser, not the wire. One run per case cannot separate effects below about 20%; the protocol's three repeats are still owed.
- The earlier first case (4.87 images/s, float64 wire, H=5) ran on a different coordinator; compare the wire and pixel cases with the 5.99 control above.

## What this stage does not measure

Start with `--synthetic`: random CIFAR-shaped images and labels, fully reproducible for a given case. `--data` optionally reads CIFAR training images from a prepared NPZ, but samples random batches with replacement across steps; it is still a throughput benchmark. It does not evaluate validation accuracy, convergence or time to a target accuracy. A proper learning study follows only after throughput and multi-round numerical checks pass. Do not label synthetic throughput as CIFAR learning quality.

The coordinator uses trusted, fixed membership; machine labels are self-reported, not hardware attestation. A lost/reloaded worker loses its private Adam moments. Failed jobs are recorded and must restart as a new case. Checkpoints contain global weights only and do not implement resume. JSON and per-operation GPU uploads/readbacks remain prototype costs.

## Historical implementation and current public helpers

The original hosted `serve`/`worker` commands and deployment instructions are
archived outside this repository. They are not supported commands in this public
release. The measurements above used that historical implementation.

Reusable saved-report analysis remains public in
`ygaraty_webgpu_research.convnet_remote_scaling` and
`ygaraty.analysis.convnet_scaling`. Install the optional helpers with
`python -m pip install -e ./research` from the repository root.

### Wire-format comparison

The historical flags below identify the measured case settings, not current
launch options.

`--wire-dtype float32` sends each round's weights, and takes each pseudo-gradient back, as base64 little-endian float32 instead of JSON float64 numbers; WebGPU and native workers follow the assignment's setting. The coordinator's weights and outer step stay float64, and plan constants (the images) are unchanged. It is a separate case setting (recorded in the report's `config.wire_dtype`): compare float32-wire cases with each other, or against float64-wire cases explicitly. On loopback with a native worker (full model, batch 8, H=5, four rounds), it cut the coordinator's sent bytes from 9.8 to 3.8 MB and received bytes from 8.4 to 2.0 MB; final weights differed from the float64-wire case by at most 3.3e-6. `--pixels uint8` (WebGPU cases) sends each local step's images as base64 uint8 pixels; the browser rebuilds the recipe's float64 `x/127.5-1` with the same operations, so losses and pseudo-gradients are bitwise identical to `--pixels float64` (tested in Node). With the float32 wire, a full-model round's download falls from 3.0 to 0.76 MB at H=5 and from 10.5 to 1.5 MB at H=20; the remainder is the compiled task lists.

## Full-model numerical gate before scaling

Run one full-model WebGPU case and one native **CPU float64** case with identical seed/data/batch/H/round settings, starting with `--replicas 1 --batch 2 --local-steps 1 --warmup-rounds 0 --rounds 3`. Compare final `weights.npz` arrays using `np.testing.assert_allclose(..., rtol=.003, atol=2e-5)`; also inspect saved losses. Preserve discrepancies and stop the scaling matrix on a failed gate. These tolerances are declared before the full-model test, not calibrated convergence evidence.

```bash
python - <<'PY'
import numpy as np
# Use the actual full-model gate output directories here.
a = np.load('runs/experiments/045/gate-cpu/weights.npz')
b = np.load('runs/experiments/045/gate-webgpu/weights.npz')
assert set(a.files) == set(b.files)
for name in a.files:
    np.testing.assert_allclose(b[name], a[name], rtol=.003, atol=2e-5)
print('Full-model multi-round checkpoint parity passed')
PY
```

The initial preparation used tiny smoke tests; the subsequent full-model gate passed as recorded in Results so far above. Even a pass checks only this small batch/round count, not all proposed batch/H settings; inspect finite losses and retain every failure during the matrix.

## Input and smoke-test controls

The historical real-data cases used prepared CIFAR training arrays; validation
and test labels were unused. Reports record the NPZ SHA-256. Do not compare
synthetic and real-data throughput as if inputs were identical.

Tiny smoke tests used two browser tabs on one host, explicitly distinguished
from physical-machine scaling. Native and WebGPU checkpoints passed the declared
tolerance after three rounds. The original setup commands are archived separately.

## Read the results

Each case produces `report.json` and `weights.npz`. The report is saved after each submitted result and at exit; interrupted/incomplete cases stay labeled accordingly. Use:

```bash
python -m ygaraty_webgpu_research.convnet_remote_scaling report \
  runs/experiments/045/webgpu-m1-h5-r1 \
  runs/experiments/045/webgpu-m2-h5-r1 \
  runs/experiments/045/cuda-f32-m1-h5-r1 \
  runs/experiments/045/cuda-f32-m2-h5-r1
```

The comparison refuses incomplete cases and unlike model/data/batch/H/seed/round/source settings. It prints images/sec and scaling relative to the matching backend's one-worker baseline. Keep native device/precision/thread settings and hardware membership consistent when interpreting scaling. The helper does not infer equivalent hardware from machine labels. Repeat all cases three times, compare median throughput, and disclose the variation and straggler times.
