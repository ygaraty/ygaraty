# Larger fused factorization: remote test procedure

The code is prepared for an external smoke test followed by a measured scaling
campaign. **Remote end-to-end execution has not yet been validated.** Local
validation uses shapes, small file fixtures, fake HTTP and independent tiny
numerical equations. Never run the coordinator, workers, `run`, or input
generation for this campaign on the simulation-only development machine.

This tests dense, full-batch SGD update throughput, not convergence or useful
recommendation quality. It does not establish Transformer scalability.

## What is now supported

- The `factorization` recipe accepts `implementation: "fused"` and `block_rows`.
  It uses the tested row-shard kernel, common factor snapshot, bounded V/loss
  reductions and once-only regularization.
- X/U/V are separate little-endian float64 row-major files. The recipe contains
  only shape, byte count, dtype and SHA-256 descriptors; filenames are
  `<sha256>.f64`. Arbitrary uploaded task graphs remain rejected by recompilation.
- Tensor upload requests carry at most 256 KiB decoded data. Finalization checks
  the full digest, and chunks must contain finite float64 values. Uploads are
  held on disk, expire after an hour, and can be retried with the same keys.
  Each tensor is capped at 256 MiB, all pending tensors at 512 MiB, and there
  are at most 32 tensor uploads. Graph bundles retain the 4 MiB cap.
- A 4,096 × 4,096 rank-32 problem with 64-row shards has 358 tasks, a roughly
  92 KiB graph bundle, 130 MiB of input tensors, and a 342.02 MiB tensor-store
  reservation. The observation matrix alone is 128 MiB. Store blocks are
  128 × 128 or smaller; kernel row shards are independent of store tiles.
- Stored payload reservations total at most 1 GiB across submitted jobs.
  This is not a process RSS limit: allow extra RAM for arrays, HTTP framing,
  Python metadata and concurrent requests. Use a dedicated coordinator with
  at least 4 GiB available RAM and sufficient temporary-disk space initially.
- External-input jobs require demo memory storage and **no `--submission-state`**.
  Existing JSON checkpoints are for small jobs. This campaign is ephemeral;
  coordinator restart loses jobs and invitations. Cancel a completed job after
  saving its reports to release its store reservation. Restart the coordinator
  between campaigns to clear the existing 16-job session limit.

The first campaign uses Python CPU workers, one task per claim, no relay,
float64 transport, and no cache-affinity scheduling. Do not compare it directly
with relay-enabled simulations as though their transport paths were identical.
An optional relay needs a separate controlled campaign and its own byte/RSS
measurements. Browser/WebGPU and heterogeneous pools are later experiments.

## Prepare the execution environment

Use the same **clean committed revision** on the coordinator, control host and
all workers. The harness rejects dirty or mismatched revisions. Commit/deploy
the prepared changes before using it. Use the same Python/NumPy environment,
fixed CPU/BLAS thread counts, and identical worker machine types for the first
campaign. Record CPU model, VM type, actual link topology, and any host/network
limits alongside the reports; identical thread counts alone do not prove
identical compute capacity. Keep the coordinator on a separate machine.

On the authorized coordinator host, clear inherited storage, dataset and relay
configuration. Explicitly select verification/demo mode. The recipe controls
whether results are numerically verified; benchmark recipes disable exhaustive
verification so the coordinator does not redo the workers' numerical work.

```bash
env -u STORE_URL -u YGARATY_STORE_URL \
    -u DATASET_URL -u YGARATY_DATASET_URL \
    -u RELAY_URL -u YGARATY_RELAY_URL \
    YGARATY_MODE=verification \
    ygaraty coordinator serve --host 127.0.0.1 --port 8090
```

Expose this through the deployment's existing authenticated HTTPS setup.
Configure the client environment's `YGARATY_ADMIN_TOKEN` privately with the
coordinator's printed admin token. Do not put tokens in shell arguments or
commit invitation files. The commands below use `https://compute.example` as
a placeholder. They provision no machines or TLS configuration.

## First gate: two-machine correctness smoke test

On the authorized control host, prepare small fixed inputs and one SGD update:

```bash
ygaraty experiment factorization_remote prepare \
  --directory runs/factorization-remote/smoke \
  --rows 32 --columns 24 --rank 4 --block-rows 8 --steps 1 --smoke
ygaraty computation validate runs/factorization-remote/smoke/plan.ygplan
ygaraty submit runs/factorization-remote/smoke/plan.ygplan \
  --input-dir runs/factorization-remote/smoke \
  --coordinator https://compute.example --idempotency-key fused-smoke-1
```

Retain the returned job ID as `JOB_ID`. Submission creates a paused job. Reuse
the same idempotency key after an uncertain submission, but use a new key for
each independent benchmark run.

Generate an invitation, preserving its sensitive JSON in a private file:

```bash
umask 077
ygaraty compile invitation JOB_ID \
  --coordinator https://compute.example --ttl-seconds 3600 > runs/invitation.json
```

Make that invitation file available securely to the two authorized worker
hosts. On each host run the following, with distinct IDs `w1` and `w2` and
distinct output paths. The wrapper fixes BLAS thread settings before NumPy is
imported, sends the capacity handshake, and waits on the paused job.

```bash
ygaraty experiment factorization_remote worker \
  --coordinator https://compute.example --job JOB_ID \
  --invitation runs/invitation.json --id w1 \
  --blas-threads 1 --cache-mib 512 --output runs/smoke-w1.json
```

Create `runs/inventory.json` on the control host, replacing values with the
actual `platform.node()` hostnames of those machines:

```json
{"w1":"worker-host-one","w2":"worker-host-two"}
```

The control host starts only once both expected IDs have checked in:

```bash
ygaraty experiment factorization_remote run \
  --coordinator https://compute.example --job JOB_ID \
  --bundle runs/factorization-remote/smoke/plan.ygplan \
  --input-dir runs/factorization-remote/smoke \
  --inventory runs/inventory.json --timeout 600 --output runs/smoke-report.json
```

The coordinator verifies task results for this small run. The controller then
fetches the final small U/V arrays and independently checks loss and one SGD
update at rtol 1e-10 / atol 1e-12. Check `smoke.passed`, zero rejection/retry
counters and actual task participation from both hosts before advancing.
Do not count this verified run in performance aggregates. The numerical result
export endpoint is admin-only and limited to small factorization jobs.

## Second gate: fixed-workload strong scaling

On the authorized control host, prepare inputs **once** and reuse the exact
bundle and tensors for every machine count and repetition:

```bash
ygaraty experiment factorization_remote prepare \
  --directory runs/factorization-remote/large \
  --rows 4096 --columns 4096 --rank 32 --block-rows 64 --steps 22 --seed 2026
```

Repeat the submission → invitation → worker → `run` procedure for **1, 2, 4,
and 8 separate worker machines**, with **three independent runs per count**.
Use a fresh job/key and fresh worker processes each time, but the same input
files. Interleave the machine counts across repetitions to expose time-of-day
effects. Keep BLAS threads, cache budget, hardware and claim size fixed.
Do not include `--smoke`; exhaustive numerical checking would distort timing.

Keep each coordinator report and every worker report. Once all reports are
saved, release a completed job's memory reservation with the existing command:

```bash
ygaraty compile cancel JOB_ID --coordinator https://compute.example
```

The runner does not automatically cancel successful jobs, preserving their
state for inspection. A failure or timeout after it starts a job saves a
failure report and attempts cancellation. If cancellation itself fails, inspect
and cancel the job explicitly. Retries/failures are excluded from the clean
throughput summary and must be reported separately.

The existing session supports 16 jobs; the smoke plus 12 benchmark runs fits.
Expired input uploads can be resubmitted. A completed idempotency key identifies
the old job, not a new repetition. The control host can inspect per-update and
worker counters with `ygaraty computation status` throughout the run.

## Offline aggregation and interpretation

Collect reports through the execution environment's normal artifact workflow.
This development machine must not initiate remote jobs or artifact sync.
Once the report files are available locally, aggregation is offline:

```bash
ygaraty experiment factorization_remote summarize \
  --runs runs/campaign/control-*.json \
  --workers runs/campaign/worker-*.json \
  --output runs/campaign/scaling.json
```

The summary requires an identical bundle/revision, a one-machine baseline,
three runs per count, equal worker thread/cache/claim/Python/NumPy settings,
and a worker report matching each expected physical hostname. It rejects
predictions, smoke tests, missing evidence, duplicate jobs, retries and failed
runs. Hostnames are operator/worker assertions, not hardware attestations;
retain the deployment inventory as supporting evidence.

Each run's steady time is the median of updates 2–21. Cold and terminal update
times and the sum of all update times are retained separately; upload, initial
store publication and worker handshake happen before the timed start gate.
For each machine count the summary reports the median/min/max across runs,
speedup T(1)/T(p), and efficiency T(1)/(p*T(p)). Three repeats provide an initial
range, not a robust confidence interval.

Coordinator reports contain per-update transfer counters and tensor-store
high-water marks. Worker reports contain host identity, fixed settings, process
peak RSS and request/compute spans. Transfer counters are payload accounting,
not all on-wire bytes; worker stages overlap and must not be added to update
wall time. Tensor-store bytes are not coordinator process RSS. Capture
coordinator/worker OS CPU, RSS and NIC counters externally for bottleneck
analysis. Avoid interpreting the short hello GFLOP/s probe as a calibrated
placement ranking.

Success at this gate means correct two-host execution and reproducible measured
multi-machine update scaling. Use a declared 80% efficiency target to evaluate
near-linear scaling; do not alter the workload to force a pass. A tuned
standalone numerical baseline, sustained time-to-quality, larger Transformers,
recovery and heterogeneous/home-network campaigns remain separate gates.
