# Capture the experiment-37 residual

Prepared locally; no DigitalOcean job, service, network probe or training run has been launched. Enable this only on the separate machines authorized to run the experiment. The local simulator workspace remains offline.

## Run design

Repeat A (corpus order, batch 64), B (sorted 3,072 padded tokens), C (sorted 6,144) with the **same** model, dataset/tokenizer, seed, precision, owner layout, 8 bands, 512 MiB task/layer budgets, 1,024 MiB output budget, 16 workers, 50 ms RTT and 100/20 Mb/s limits. Keep 22 updates and checkpoint-every 1,000. Capture all updates; compare steady **2–21**, startup 1 and final publication 22 separately. Preserve the original runner's timestamps and its definition of seconds/update.

For each configuration, use at least three independent repetitions. Alternate the run order to expose host/time effects. Include otherwise identical capture-off/capture-on pairs on the same allocation, with identical batches and controlled initial state, to measure instrumentation overhead. Keep one whole repeat per configuration held out from fitting. Record restarts, failed attempts, owner replacements and warm-cache differences; do not silently remove slow updates.

The package now records substantial per-task detail. Its overhead is **not yet measured** on DigitalOcean. Bounded asynchronous writing avoids synchronous disk writes on the task path, but snapshot allocation, asynchronous plan hashing and JSON writing consume CPU. Do not subtract an assumed constant overhead or refit the model until paired runs quantify it.

## Enable and collect

On the coordinator and **each Python worker**, set the following environment variables in the existing external runner before those processes start:

- `YGARATY_CAPTURE_DIR`: a fresh local directory for this run/repetition/host, for example `/var/tmp/ygaraty-capture/B-repeat-01/worker-03`.
- `YGARATY_CAPTURE_REVISION`: the exact source revision deployed, with a dirty-patch identifier if applicable.

No application launch command is supplied here to run on the simulation machine. Keep existing worker flags, BLAS threads and network limits unchanged. The Go relay and PostgreSQL do not use these variables; collect their OS metrics separately below. Do not add hello probes or change placement just to gather metadata: preserve the original run behavior.

Each Python process creates a unique `HOST-PID-CAPTURE.jsonl`. Retain **every coordinator and worker file**, including workers that failed or restarted. Allow processes to finish and close cleanly after all work completes. A clean last record is `capture_end` with `dropped: 0` and `failed: false`. Truncation, a missing footer or dropped records makes the capture incomplete. Killing a process can leave buffered rows unwritten. A long-lived coordinator must be cleanly stopped after the campaign to seal its file; ongoing files are provisional.

The capture is opt-in and writes only local files. It adds a generated `X-Ygaraty-Trace` request identifier for correlation when enabled. It does not log authorization headers, URLs, command lines, request/response bodies, exception messages, tensor values or raw lease tokens. Attempts use SHA-256 correlation hashes. Plans may contain **dataset token IDs in task attributes**, and batch records contain example IDs and lengths; keep these artifacts under the same access controls as the dataset. Existing runner/job logs have their own redaction requirements. Do not copy `job.json`, credentials or an unrestricted environment dump.

## What the new code captures

| Evidence | Events/fields | What it distinguishes |
| --- | --- | --- |
| Exact workload | `batch`, `plan`: example IDs/cursor/epoch, dataset/tokenizer hashes, ordered source/target lengths, pair/band counts, real/padded tokens, padded shapes, hash of source/decoder/label token arrays; complete shape/attribute graph with checksum, task budgets, ownership, publication flag | Different batches, padding, graph versions or owner states versus timing-model error |
| Update boundaries | `update_ready`, `update_tasks_done`, `coordinator.make_step`, `coordinator.finish_step`, `coordinator.finish_step_io` | Graph planning and input seeding, task execution, update bookkeeping and publication/export |
| Worker kernels | `worker.kernel`: task, hashed attempt, owner/layer/band, operation, input/output shapes, cost-model work and extra-work, monotonic wall and thread CPU duration | Hardware-specific kernel costs and shape scaling; CPU descheduling versus active kernel work |
| Conversion and input handling | `worker.request_encode`, `worker.result_encode`, `worker.reply_decode_cache`, `worker.fetch_assemble`, `relay.assemble_decode`, `relay.encode`, `relay.hash` | JSON/frame overhead, cache work, bf16/f32 conversion, copies and hashing outside pure kernels |
| Upload concurrency | `worker.upload_queue`, `worker.upload`, `worker.post_queue`, `relay.pool_queue` | Waiting for the two result upload slots, the relay client's 16 block-upload threads, or a result-post batch |
| HTTP | `worker.claim_http`, `worker.result_http`, `worker.post_http`, `relay.http`; nested `http.send`, `http.wait_headers`, `http.read`; body byte counts, status, generated request ID | Sending/receiving versus response wait; repeated attempts count separately |
| Coordinator | `server.request`, `server.request_read`, `server.response_total`, `server.response_write`, `coordinator.claim`, `claim_wait`, `claim_now`, `push_relayed`, `record_result`, `commit`, `flush`, `lock_wait`, `lock_hold` | Held claims, lock contention, scheduler/response work, pushed inputs and result acceptance |
| Scheduler lifecycle | `scheduler.ready`, `assigned`, `sent`, `completed`, `retry`, `ignored`, `owner_assigned`; hashed attempt/worker/task, update | Ready-to-assigned queueing, assigned-to-commit latency, retries and owner placement |
| Storage | `store.read/put/put_remote/put_block/publish/lease/release/locate/assemble/transpose/cleanup_attempt` | Public store/client call time, including its database/I/O work; not pure PostgreSQL execution time |
| Physical bytes | `relay.encode.payload_bytes`, `relay.download_bytes` payload versus framing, HTTP request/response body bytes, worker pushed-payload bytes | bf16 = 2 bytes, f32 = 4 and f64 = 8; avoids counting float64 compute arrays as wire bytes |
| Provenance/resources | `capture_start`, `libraries`, `worker_config`, approximately 1 Hz `process_sample`, clean footer | Source-tree hash, supplied revision, Python/NumPy/BLAS/thread settings; process CPU, RSS high-water mark, faults, context switches and capture loss |

Scope: detailed upload-queue instrumentation targets the **eager-claims StreamPoster path used by experiment 37**. Legacy coalesced posters have HTTP/relay spans but not equivalent per-result queue spans. Warmup/hello probes are not traced by the generic task request path. The Go relay's internal phases are not instrumented; Python relay HTTP timing plus OS metrics bound that remaining uncertainty.

All span durations use process-local monotonic nanoseconds and thread CPU clocks. UTC anchors aid alignment but are not a guarantee of synchronized hosts. Join workers to coordinator lifecycle events by `(task, attempt_hash)` and HTTP requests by generated request ID. Scheduler events carry the authoritative job/update; a long-held request's update label is its entry update and may span a transition. Per-thread order and request/attempt joins should be used before wall-clock subtraction across hosts.

Spans are nested and concurrent: **never sum them to explain the update wall time**. HTTP wait includes network, relay/server work and queueing. Store spans include client and DB work. `thread_cpu_ns` does not measure other BLAS threads; verify single-thread BLAS from provenance. Queue waits can overlap useful computation. Use the captured DAG and lifecycle events to find the delayed critical-path tasks, then inspect their local worker/server spans. Keep CPU descheduling and serialization separate from kernel-fit coefficients.

## Retain alongside the traces

Fill `run-manifest.example.json` per run, and preserve:

1. **Deployment and numerical configuration:** exact commit, actual source patch/archive (the tree hash alone cannot reconstruct it), Python/NumPy/BLAS versions, CPU model/flags, vCPU/memory/SKU, kernel, BLAS thread settings, relay binary hash/version, PostgreSQL version/configuration. Record flags via an allowlist, with all tokens and connection credentials omitted. Dataset and tokenizer checksums must match captured batch provenance.
2. **Original run outputs:** sanitized runner logs, all 22 per-update times/tokens/pairs, coordinator metrics/events/task-times if already produced, final status, retries and errors. Preserve the timer boundaries, units and unrounded values; don't retain only the 20-update mean. Startup/load and final publication have their own records.
3. **One-second OS time series on every worker:** CPU user/system/idle/iowait/**steal**, run queue, per-process CPU/RSS/context switches, disk I/O and interface byte/error/drop counters. On the server capture coordinator, Go relay, PostgreSQL processes and latency proxy separately. Use preinstalled tools such as `pidstat`, `mpstat`, `iostat`, `sar` or equivalent `/proc` collectors in the external runner. No tools are installed by this patch. Record collector versions and overhead.
4. **Network and clock state:** exact latency-proxy implementation/configuration/revision, per-direction limit, delay, buffer/concurrency policy, interface/offload settings; shaping counters before/after, retransmissions, drops and any existing measured RTT/throughput. Save clock synchronization offset/error at start/end (for example chrony tracking). Collect separate calibration probes outside the timed training interval only if authorized there. The configured 50 ms / 100/20 Mb/s is not itself a measurement of effective throughput.
5. **Database/storage:** aggregate PostgreSQL wait-event/activity snapshots and before/after statistics for reads, writes, temporary bytes and transaction counts. Capture query fingerprints/aggregate timings if already available, avoiding full SQL values. This distinguishes server scheduling from database latency without placing SQL text or secrets into the application trace.

Use one manifest and capture directory per configuration/repetition. Host metrics and original runner logs live alongside, not inside the JSONL-only directory passed to the extractor.

## Offline extraction and next comparison

After the files are copied here through the user's existing process, this command only reads local captures and writes a new local output directory:

```bash
ygaraty experiment calibration_capture \
  /absolute/path/to/capture-jsonl-only \
  --output /absolute/path/to/new-derived-report \
  --expected-workers 16 --expected-updates 22
```

The extractor checks schema, identities, graph hashes, clean footers, per-job worker counts, durations, and one plan/batch record per update with complete boundaries. Invalid/incomplete captures produce `report.json`, exit 2 and **no calibration samples or plans**. Completeness checks do not establish correct clocks, exact workload equivalence or calibrated costs.

A complete capture exports replayable `plans/*.json`, compute-only `kernel-samples.json` (operation/work/seconds plus shapes and provenance links), and `report.json` with source-file checksums, manifest records, update intervals and stage summaries. `ready_to_tasks_done_s` excludes planning; `completion_interval_s` includes the transition after the previous update. Neither is silently equated to the runner's original KPI. End-publication I/O remains a separate span. Kernel samples can contain retried attempts; join accepted attempts before fitting, and compare attempts separately when explaining retry overhead. Samples preserve `extra_work`; fused tasks with nonempty extra work must be modeled jointly or excluded from a simple single-operation fit.

First verify batches and timers against the original KPI. Replay the exact captured plans using the existing reference suite, matching startup/publication flags and owner state. Then fit **separate c-2 kernel costs**, retaining shape/work-range warnings, on training repeats only. Evaluate on held-out repeats before changing simulator defaults. Compare serialization, transfer, scheduler and storage residuals on the critical path; do not absorb all overhead into a global kernel multiplier. Until then, the hypothesis that kernels are 25% slower remains unproven.

## Local verification

Eighteen instrumentation tests use synthetic metadata, fake HTTP connections and blocked socket connects. They check cross-thread context, physical bf16 sizes, eager upload/post queues, request correlation, secret omission, replayable snapshots, bounded-queue loss and incomplete-capture rejection. The ten existing simulator/reference tests pass, and the shape-only planner/CLI test also passes with capture enabled. No live coordinator, relay, worker training loop or DigitalOcean run was started. Runtime overhead and real-host behavior remain to be validated externally.

See [the offline review](REVIEW.md) for fixed bugs, synthetic timing measurements, raw results and remaining DigitalOcean validation requirements.

## Transport and quality follow-up

See [NEXT-RUNS.md](NEXT-RUNS.md) for the 50/53/54 capture matrix, new Python
and optional browser/CDP instrumentation, whole-run kernel refits and immutable
quality snapshots. Experiment 37 remains validation-only.
